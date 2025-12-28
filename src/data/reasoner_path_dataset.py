from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from src.utils.llm_prompting import build_path_prompt
from src.utils.text_utils import count_tokens

from .answer_utils import normalize_answer_texts
from .g_agent_dataset import GAgentSample, load_g_agent_samples

DIRECTION_FORWARD = 0
DIRECTION_BACKWARD = 1


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
        raise FileNotFoundError(f"vocab parquet not found: {path}")
    df = pd.read_parquet(path)
    if key not in df.columns or val not in df.columns:
        raise ValueError(f"vocab parquet missing required columns: {key}, {val}")
    if df.empty:
        raise ValueError(f"vocab parquet is empty: {path}")
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
        g_agent_path: Optional[str] = None,
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
        if relation_vocab_path is None:
            raise ValueError("relation_vocab_path is required.")
        self.relation_vocab_path = Path(relation_vocab_path).expanduser().resolve()
        if g_agent_path is None:
            raise ValueError("g_agent_path is required.")
        self.g_agent_path = Path(g_agent_path).expanduser().resolve()
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

        self._ent_map = _load_parquet_map(self.entity_vocab_path, "entity_id", "label")
        self._rel_map = _load_parquet_map(self.relation_vocab_path, "relation_id", "label")

        if not self.g_agent_path.exists():
            raise FileNotFoundError(f"g_agent_path not found: {self.g_agent_path}")
        samples = load_g_agent_samples(self.g_agent_path, drop_unreachable=False)
        self._g_agent_map = {sample.sample_id: sample for sample in samples}

        self._answer_map = self._load_answer_map(self.questions_path, self.answer_text_field)

        if self.eval_cache_path.suffix == ".jsonl":
            payload = _load_jsonl(self.eval_cache_path)
        elif self.eval_cache_path.suffix == ".json":
            payload = json.loads(self.eval_cache_path.read_text(encoding="utf-8"))
        else:
            payload = _load_torch(self.eval_cache_path)
        if not isinstance(payload, dict) or "samples" not in payload:
            raise ValueError("eval_gflownet cache must be a dict with 'samples'.")
        self.samples = payload["samples"]
        if not isinstance(self.samples, list):
            raise ValueError("eval_gflownet cache 'samples' must be a list.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw = self.samples[idx]
        if "sample_id" not in raw:
            raise ValueError(f"sample_id missing for record index {idx}")
        sample_id = raw["sample_id"]
        if "question" not in raw:
            raise ValueError(f"question missing for sample_id={sample_id}")
        question = raw["question"]
        if sample_id not in self._g_agent_map:
            raise ValueError(f"g_agent sample missing for sample_id={sample_id}")
        g_sample = self._g_agent_map[sample_id]
        answer_ids = g_sample.answer_entity_ids.view(-1).tolist()
        answers = self._resolve_answers(sample_id)
        if "candidate_chains" not in raw:
            raise ValueError(f"candidate_chains missing for sample_id={sample_id}")
        candidate_chains = raw["candidate_chains"]
        if not isinstance(candidate_chains, list):
            raise ValueError(f"candidate_chains must be a list for sample_id={sample_id}")
        chains_full = self._prepare_chains(candidate_chains, keep_edges=True)
        hit_set, hit_vis, retrieved_edge_ids, visible_edge_ids = self._compute_hit_flags(
            chains_full,
            g_sample=g_sample,
            sample_id=sample_id,
        )
        chains = self._strip_chain_edges(chains_full)
        user_prompt = build_path_prompt(
            question=question,
            chains=chains,
            limit=self.max_chains_per_sample,
            include_meta=self.include_meta,
            instruction=self.user_instruction,
        )
        evidence_lines = [c["chain_text"] for c in chains[: self.max_chains_per_sample]]

        return {
            "id": sample_id,
            "question": question,
            "answers": answers,
            "answer_entity_ids": answer_ids,
            "paths": chains,
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "prompt_tag": self.prompt_tag,
            "retrieved_edge_ids": retrieved_edge_ids,
            "visible_edge_ids": visible_edge_ids,
            "gt_path_edge_local_ids": raw["gt_path_edge_local_ids"],
            "hit_set": hit_set,
            "hit_vis": hit_vis,
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
        return normalize_answer_texts(raw, field_name=self.answer_text_field, sample_id=sample_id)

    @staticmethod
    def _strip_chain_edges(chains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        stripped: List[Dict[str, Any]] = []
        for chain in chains:
            stripped.append(
                {
                    "chain_text": chain["chain_text"],
                    "length": int(chain["length"]),
                    "frequency": int(chain["frequency"]),
                }
            )
        return stripped

    def _build_candidate_chains_from_rollouts(
        self,
        rollouts: List[Dict[str, Any]],
        *,
        g_sample: GAgentSample,
        sample_id: str,
    ) -> List[Dict[str, Any]]:
        heads = g_sample.edge_head_locals.view(-1).tolist()
        tails = g_sample.edge_tail_locals.view(-1).tolist()
        relations = g_sample.edge_relations.view(-1).tolist()
        node_entity_ids = g_sample.node_entity_ids.view(-1).tolist()
        num_edges = len(heads)

        chain_stats: Dict[tuple, Dict[str, Any]] = {}
            for ridx, rollout in enumerate(rollouts):
                if not isinstance(rollout, dict):
                    raise ValueError(f"rollout must be dict (sample_id={sample_id}, rollout_index={ridx})")
            if "edge_ids" not in rollout or "directions" not in rollout:
                raise ValueError(f"rollout missing edge_ids/directions (sample_id={sample_id}, rollout_index={ridx})")
            edge_ids_raw = rollout["edge_ids"]
            directions_raw = rollout["directions"]
            if not isinstance(edge_ids_raw, list) or not isinstance(directions_raw, list):
                raise ValueError(f"rollout missing edge_ids/directions (sample_id={sample_id}, rollout_index={ridx})")
            if len(edge_ids_raw) != len(directions_raw):
                raise ValueError(
                    f"edge_ids/directions length mismatch (sample_id={sample_id}, rollout_index={ridx})"
                )

            sig = []
            edges: List[Dict[str, Any]] = []
            edge_local_ids: List[int] = []
            for eidx, (edge_id_raw, direction_raw) in enumerate(zip(edge_ids_raw, directions_raw)):
                try:
                    edge_id = int(edge_id_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"edge_ids[{eidx}] must be int (sample_id={sample_id}, rollout_index={ridx})"
                    ) from exc
                if edge_id < 0 or edge_id >= num_edges:
                    raise ValueError(
                        f"edge_id out of range for sample_id={sample_id}: {edge_id} not in [0, {num_edges}). "
                        "eval_gflownet likely emitted batch-global edge ids; regenerate with local edge ids."
                    )
                try:
                    direction = int(direction_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"directions[{eidx}] must be int (sample_id={sample_id}, rollout_index={ridx})"
                    ) from exc
                if direction not in (DIRECTION_FORWARD, DIRECTION_BACKWARD):
                    raise ValueError(
                        f"directions[{eidx}] invalid (sample_id={sample_id}, rollout_index={ridx}): {direction}"
                    )

                head_local = int(heads[edge_id])
                tail_local = int(tails[edge_id])
                rel_id = int(relations[edge_id])
                if direction == DIRECTION_FORWARD:
                    src_local, dst_local = head_local, tail_local
                else:
                    src_local, dst_local = tail_local, head_local

                src_entity_id = int(node_entity_ids[src_local])
                dst_entity_id = int(node_entity_ids[dst_local])
                head_entity_id = int(node_entity_ids[head_local])
                tail_entity_id = int(node_entity_ids[tail_local])
                edge_local_ids.append(edge_id)
                sig.append((src_entity_id, rel_id, dst_entity_id))
                edges.append(
                    {
                        "edge_id": edge_id,
                        "head_entity_id": head_entity_id,
                        "tail_entity_id": tail_entity_id,
                        "relation_id": rel_id,
                        "src_entity_id": src_entity_id,
                        "dst_entity_id": dst_entity_id,
                        "src_node_local": src_local,
                        "dst_node_local": dst_local,
                    }
                )

            sig_tuple = tuple(sig)
            if not sig_tuple:
                continue
            stat = chain_stats.setdefault(
                sig_tuple,
                {
                    "frequency": 0,
                    "from_rollouts": set(),
                    "example_edges": edges,
                    "edge_local_ids": edge_local_ids,
                },
            )
            stat["frequency"] += 1
            stat["from_rollouts"].add(ridx)

        candidates: List[Dict[str, Any]] = []
        for sig, stat in chain_stats.items():
            edges = stat["example_edges"]
            chain_text = " -> ".join(self._fmt_edge(e) for e in edges)
            candidates.append(
                {
                    "signature": sig,
                    "length": len(edges),
                    "frequency": stat["frequency"],
                    "from_rollouts": sorted(stat["from_rollouts"]),
                    "chain_edges": edges,
                    "edge_local_ids": stat["edge_local_ids"],
                    "chain_text": chain_text,
                }
            )
        return candidates

    def _compute_hit_flags(
        self,
        chains: List[Dict[str, Any]],
        *,
        g_sample: Optional[GAgentSample],
        sample_id: str,
    ) -> Tuple[bool, bool, List[int], List[int]]:
        if g_sample is None:
            raise ValueError(f"g_agent sample missing for sample_id={sample_id}; cannot compute hit_set/hit_vis.")
        if not chains:
            return False, False, [], []
        if not any("chain_edges" in chain for chain in chains):
            raise ValueError(f"candidate_chains missing chain_edges for sample_id={sample_id}")

        pair_map: Dict[Tuple[int, int], int] = {}
        for s, a, length in zip(
            g_sample.pair_start_node_locals.view(-1).tolist(),
            g_sample.pair_answer_node_locals.view(-1).tolist(),
            g_sample.pair_shortest_lengths.view(-1).tolist(),
        ):
            pair_map[(int(s), int(a))] = int(length)

        entity_to_local = {int(eid): idx for idx, eid in enumerate(g_sample.node_entity_ids.view(-1).tolist())}

        hit = False
        for chain in chains:
            if self._is_shortest_chain(chain, pair_map, entity_to_local):
                hit = True
                break

        edge_ids: List[int] = []
        for chain in chains:
            if "edge_local_ids" not in chain:
                raise ValueError(f"edge_local_ids missing for sample_id={sample_id}")
            edge_local_ids = chain["edge_local_ids"]
            edge_ids.extend(int(eid) for eid in edge_local_ids)

        return hit, hit, edge_ids, edge_ids

    @staticmethod
    def _is_shortest_chain(
        chain: Dict[str, Any],
        pair_map: Dict[Tuple[int, int], int],
        entity_to_local: Dict[int, int],
    ) -> bool:
        if "chain_edges" not in chain:
            raise ValueError("chain_edges missing from chain.")
        edges = chain["chain_edges"]
        if not edges:
            return False
        if "src_node_local" not in edges[0] or "dst_node_local" not in edges[-1]:
            raise ValueError("chain_edges missing src_node_local/dst_node_local.")
        start_local = edges[0]["src_node_local"]
        end_local = edges[-1]["dst_node_local"]
        if "length" not in chain:
            raise ValueError("chain missing length.")
        length = int(chain["length"])
        return pair_map.get((int(start_local), int(end_local))) == int(length)

    def _prepare_chains(self, chains: List[Dict[str, Any]], *, keep_edges: bool = False) -> List[Dict[str, Any]]:
        def _key(chain: Dict[str, Any]) -> Tuple:
            keys = []
            for field in self.sort_by:
                desc = field.startswith("-")
                name = field[1:] if desc else field
                if name not in chain:
                    raise ValueError(f"chain missing sort key '{name}'")
                val = chain[name]
                keys.append((-val if desc else val))
            return tuple(keys)

        filtered = []
        for c in chains:
            if "length" not in c:
                raise ValueError("chain missing length")
            length = int(c["length"])
            if length < self.min_chain_length:
                continue
            if self.max_chain_length is not None and length > self.max_chain_length:
                continue
            if "chain_text" not in c:
                raise ValueError("chain missing chain_text")
            chain_text = c["chain_text"]
            if "frequency" not in c:
                raise ValueError("chain missing frequency")
            entry = {
                "chain_text": chain_text,
                "length": length,
                "frequency": int(c["frequency"]),
            }
            if keep_edges:
                if "chain_edges" not in c:
                    raise ValueError("chain missing chain_edges")
                if "edge_local_ids" not in c:
                    raise ValueError("chain missing edge_local_ids")
                entry["chain_edges"] = c["chain_edges"]
                entry["edge_local_ids"] = c["edge_local_ids"]
            filtered.append(entry)

        filtered.sort(key=_key)
        return filtered[: self.max_chains_per_sample]

    def _validate_manifest(self, data_path: Path) -> None:
        manifest_path = data_path.with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing eval_gflownet manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "artifact" not in manifest:
            raise ValueError("eval_gflownet manifest missing artifact.")
        if manifest["artifact"] != self.artifact_name:
            raise ValueError(
                f"eval_gflownet manifest artifact mismatch: expected '{self.artifact_name}', "
                f"got '{manifest['artifact']}'."
            )
        if "schema_version" not in manifest:
            raise ValueError("eval_gflownet manifest missing schema_version.")
        if int(manifest["schema_version"]) != self.schema_version:
            raise ValueError(
                f"eval_gflownet manifest schema_version mismatch: expected {self.schema_version}, "
                f"got {manifest['schema_version']}."
            )
        if "file" not in manifest:
            raise ValueError("eval_gflownet manifest missing file.")
        if manifest["file"] != data_path.name:
            raise ValueError(
                f"eval_gflownet manifest file mismatch: expected '{data_path.name}', "
                f"got '{manifest['file']}'."
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
                raise ValueError("edge missing entity id for text lookup")
            val_int = int(val_id)
            if val_int not in self._ent_map:
                raise ValueError(f"entity id {val_int} missing from vocab map")
            return self._ent_map[val_int]

        # 必须按 agent 实际行走方向构造文本：若逆着图中 head->tail 行走，
        # 需要使用 src_entity_id/dst_entity_id 交换方向，避免语义错误。
        if "src_entity_id" not in e or "dst_entity_id" not in e:
            raise ValueError("edge missing src_entity_id/dst_entity_id")
        h = _txt(e.get("src_text"), e["src_entity_id"])
        t = _txt(e.get("dst_text"), e["dst_entity_id"])
        r = e.get("relation_text")
        if r is None:
            if "relation_id" not in e:
                raise ValueError("edge missing relation_id/relation_text")
            rid = int(e["relation_id"])
            if rid not in self._rel_map:
                raise ValueError(f"relation id {rid} missing from vocab map")
            r = self._rel_map[rid]
        return f"{h} -[{r}]-> {t}"


__all__ = ["ReasonerPathDataset"]
