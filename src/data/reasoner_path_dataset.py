from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from src.utils.llm_prompting import build_path_prompt
from src.utils.text_utils import count_tokens


def _load_torch(path: Path) -> Any:
    """Torch load with weights_only guard (torch>=2.6 default)."""
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
        if isinstance(record, dict):
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _load_parquet_map(path: Path, key: str, val: str) -> Dict[int, str]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pandas is required to read vocab parquet files.") from exc
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if key not in df.columns or val not in df.columns:
        return {}
    return dict(zip(df[key].astype(int), df[val].astype(str)))


class ReasonerPathDataset(Dataset):
    """
    Dataset to serve GFlowNet eval rollouts (<split>.jsonl/.pt) to an LLM-friendly prompt format.

    Produces dict with:
      - id, question
      - paths (filtered candidate_chains)
      - system_prompt, user_prompt (built via build_path_prompt)
    """

    def __init__(
        self,
        *,
        eval_cache_path: str,
        entity_vocab_path: str,
        relation_vocab_path: Optional[str] = None,
        questions_path: str,
        answer_text_field: str = "answer_texts",
        artifact_name: str = "eval_gflownet",
        schema_version: int = 1,
        max_chains_per_sample: int = 10,
        min_chain_length: int = 1,
        max_chain_length: Optional[int] = None,
        include_meta: bool = True,
        sort_by: Sequence[str] = ("-frequency", "-length"),
        system_prompt: str,
        user_instruction: str,
        prompt_tag: str = "paths",
    ) -> None:
        super().__init__()
        self.eval_cache_path = Path(eval_cache_path).expanduser().resolve()
        self.entity_vocab_path = Path(entity_vocab_path).expanduser().resolve()
        self.relation_vocab_path = Path(relation_vocab_path).expanduser().resolve() if relation_vocab_path else None
        self.questions_path = Path(questions_path).expanduser().resolve()
        self.answer_text_field = answer_text_field
        self.artifact_name = str(artifact_name).strip()
        if not self.artifact_name:
            raise ValueError("artifact_name must be a non-empty string.")
        self.schema_version = int(schema_version)
        if self.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer.")
        self.max_chains_per_sample = int(max_chains_per_sample)
        self.min_chain_length = int(min_chain_length)
        self.max_chain_length = int(max_chain_length) if max_chain_length is not None else None
        self.include_meta = bool(include_meta)
        self.sort_by = self._validate_sort_by(list(sort_by))
        self.system_prompt = system_prompt
        self.user_instruction = user_instruction
        self.prompt_tag = prompt_tag

        if not self.eval_cache_path.exists():
            self._raise_missing_eval_gflownet()
        self._validate_manifest(self.eval_cache_path)
        if not self.entity_vocab_path.exists():
            raise FileNotFoundError(f"entity_vocab_path not found: {self.entity_vocab_path}")
        if not self.questions_path.exists():
            raise FileNotFoundError(f"questions_path not found: {self.questions_path}")

        self._ent_map: Dict[int, str] = _load_parquet_map(self.entity_vocab_path, "entity_id", "label") or _load_parquet_map(
            self.entity_vocab_path, "embedding_id", "label"
        )
        self._rel_map: Dict[int, str] = {}
        if self.relation_vocab_path is not None:
            self._rel_map = _load_parquet_map(self.relation_vocab_path, "relation_id", "label")

        self._answer_map = self._load_answer_map(self.questions_path, self.answer_text_field)

        if self.eval_cache_path.suffix == ".jsonl":
            payload = _load_jsonl(self.eval_cache_path)
        elif self.eval_cache_path.suffix == ".json":
            payload = json.loads(self.eval_cache_path.read_text(encoding="utf-8"))
        else:
            payload = _load_torch(self.eval_cache_path)
        self.samples: List[Dict[str, Any]] = payload["samples"] if isinstance(payload, dict) else payload
        if not isinstance(self.samples, list):
            raise ValueError("eval_gflownet cache format not recognized: expected list or dict with 'samples'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw = self.samples[idx]
        sample_id = raw.get("sample_id", f"sample-{idx}")
        question = raw.get("question", "")
        answer_ids: List[int] = []
        answers = self._resolve_answers(sample_id)

        chains = self._prepare_chains(raw.get("candidate_chains") or [])
        user_prompt = build_path_prompt(
            question=question,
            chains=chains,
            limit=self.max_chains_per_sample,
            include_meta=self.include_meta,
            instruction=self.user_instruction,
        )
        evidence_lines = [c.get("chain_text", "") for c in chains[: self.max_chains_per_sample]]

        return {
            "id": sample_id,
            "question": question,
            "answers": answers,
            "answer_entity_ids": answer_ids,
            "paths": chains,
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "prompt_tag": self.prompt_tag,
            "retrieved_edge_ids": [],
            "visible_edge_ids": [],
            "gt_path_edge_local_ids": raw.get("gt_path_edge_local_ids", []),
            "evidence_token_count": count_tokens("\n".join(evidence_lines)),
            "prompt_token_count": count_tokens(f"{self.system_prompt}\n{user_prompt}"),
            "token_budget": None,
            "evidence_truncated": False,
        }

    @staticmethod
    def _load_answer_map(path: Path, answer_text_field: str) -> Dict[str, Any]:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pandas is required to read questions parquet files.") from exc
        df = pd.read_parquet(path)
        if "question_uid" not in df.columns:
            raise ValueError("questions.parquet missing required column 'question_uid'.")
        if answer_text_field not in df.columns:
            raise ValueError(f"questions.parquet missing required column '{answer_text_field}'.")
        return {str(row.question_uid): getattr(row, answer_text_field) for row in df.itertuples()}

    def _resolve_answers(self, sample_id: str) -> List[str]:
        if sample_id not in self._answer_map:
            raise ValueError(f"answers missing for sample_id={sample_id}")
        raw = self._answer_map[sample_id]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError(f"{self.answer_text_field} must be a non-empty list (sample_id={sample_id})")
        answers: List[str] = []
        for idx, ans in enumerate(raw):
            if not isinstance(ans, str):
                raise ValueError(
                    f"{self.answer_text_field}[{idx}] must be string (sample_id={sample_id}), got {type(ans).__name__}"
                )
            text = ans.strip()
            if not text:
                raise ValueError(f"{self.answer_text_field}[{idx}] is empty (sample_id={sample_id})")
            answers.append(text)
        return answers

    def _prepare_chains(self, chains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _key(chain: Dict[str, Any]) -> Tuple:
            keys = []
            for field in self.sort_by:
                desc = field.startswith("-")
                name = field[1:] if desc else field
                val = chain.get(name, 0)
                keys.append((-val if desc else val))
            return tuple(keys)

        filtered = []
        for c in chains:
            length = int(c.get("length", 0))
            if length < self.min_chain_length:
                continue
            if self.max_chain_length is not None and length > self.max_chain_length:
                continue
            chain_text = c.get("chain_text")
            if chain_text is None and c.get("chain_edges"):
                chain_text = " -> ".join(self._fmt_edge(e) for e in c["chain_edges"])
            filtered.append(
                {
                    "chain_text": chain_text or "",
                    "length": length,
                    "frequency": int(c.get("frequency", 0)),
                }
            )

        filtered.sort(key=_key)
        return filtered[: self.max_chains_per_sample]

    def _validate_manifest(self, data_path: Path) -> None:
        manifest_path = data_path.with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing eval_gflownet manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact") != self.artifact_name:
            raise ValueError(
                f"eval_gflownet manifest artifact mismatch: expected '{self.artifact_name}', "
                f"got '{manifest.get('artifact')}'."
            )
        if int(manifest.get("schema_version", -1)) != self.schema_version:
            raise ValueError(
                f"eval_gflownet manifest schema_version mismatch: expected {self.schema_version}, "
                f"got {manifest.get('schema_version')}."
            )
        if manifest.get("file") != data_path.name:
            raise ValueError(
                f"eval_gflownet manifest file mismatch: expected '{data_path.name}', "
                f"got '{manifest.get('file')}'."
            )

    def _raise_missing_eval_gflownet(self) -> None:
        split = self.eval_cache_path.stem
        base_dir = self.eval_cache_path.parent
        legacy_candidates = [
            base_dir / f"{split}_eval_gflownet.jsonl",
            base_dir / f"{split}_gflownet_eval.jsonl",
        ]
        existing = [str(p) for p in legacy_candidates if p.exists()]
        if existing:
            raise FileNotFoundError(
                "eval_gflownet artifact naming mismatch. "
                f"Expected {self.eval_cache_path} but found legacy file(s): {existing}. "
                "Rename them to '<split>.jsonl' or re-run eval_gflownet."
            )
        raise FileNotFoundError(f"eval_gflownet cache not found: {self.eval_cache_path}")

    @staticmethod
    def _validate_sort_by(sort_by: List[str]) -> List[str]:
        allowed = {"frequency", "length"}
        normalized: List[str] = []
        for field in sort_by:
            if not field:
                continue
            desc = field.startswith("-")
            name = field[1:] if desc else field
            if name not in allowed:
                raise ValueError(f"sort_by contains disallowed field '{name}'. Allowed: {sorted(allowed)}")
            normalized.append(f"-{name}" if desc else name)
        if not normalized:
            raise ValueError(f"sort_by must include at least one of {sorted(allowed)}.")
        return normalized

    def _fmt_edge(self, e: Dict[str, Any]) -> str:
        def _txt(val_text: Any, val_id: Any) -> str:
            if val_text is not None:
                return str(val_text)
            if val_id is None:
                return "UNK"
            return self._ent_map.get(int(val_id), str(val_id))

        # 必须按 agent 实际行走方向构造文本：若逆着图中 head->tail 行走，
        # 需要使用 src_entity_id/dst_entity_id 交换方向，避免语义错误。
        h = _txt(e.get("src_text"), e.get("src_entity_id"))
        t = _txt(e.get("dst_text"), e.get("dst_entity_id"))
        if h == "UNK" and t == "UNK":
            h = _txt(e.get("head_text"), e.get("head_entity_id"))
            t = _txt(e.get("tail_text"), e.get("tail_entity_id"))
        r = e.get("relation_text")
        if r is None:
            rid = e.get("relation_id")
            r = self._rel_map.get(int(rid), str(rid))
        return f"{h} -[{r}]-> {t}"


__all__ = ["ReasonerPathDataset"]
