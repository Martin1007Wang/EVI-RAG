from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .reasoner_path_dataset import ReasonerPathDataset


class _ListDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


class ReasonerPathDataModule(LightningDataModule):
    """
    DataModule to serve persisted GFlowNet eval outputs to an LLM generation module.

    Produces batches of dicts with system_prompt/user_prompt/answers for predict().
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
        dataset: str = "webqsp",
        split: str = "test",
        prompt_tag: str = "paths",
        artifact_name: str = "eval_gflownet",
        schema_version: int = 1,
        max_chains_per_sample: int = 10,
        min_chain_length: int = 1,
        max_chain_length: Optional[int] = None,
        include_meta: bool = True,
        sort_by: Optional[list[str]] = None,
        system_prompt: str = "You are a concise QA agent. Use the provided paths as evidence and return JSON answers.",
        user_instruction: str = (
            "Copy all answer entities exactly as they appear in the paths (no paraphrase or alias). "
            "Return JSON only: {\"answers\": [\"<entity>\", ...]}. Use [] if the answer is absent."
        ),
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.eval_cache_path = Path(eval_cache_path).expanduser().resolve()
        self.entity_vocab_path = Path(entity_vocab_path).expanduser().resolve()
        self.relation_vocab_path = Path(relation_vocab_path).expanduser().resolve() if relation_vocab_path else None
        self.g_agent_path = Path(g_agent_path).expanduser().resolve() if g_agent_path else None
        self.questions_path = Path(questions_path).expanduser().resolve()
        self.answer_text_field = answer_text_field
        self.dataset = dataset
        self.split = split
        self.prompt_tag = prompt_tag
        self.artifact_name = str(artifact_name).strip()
        if not self.artifact_name:
            raise ValueError("artifact_name must be a non-empty string.")
        self.schema_version = int(schema_version)
        if self.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer.")
        self.max_chains_per_sample = max_chains_per_sample
        self.min_chain_length = min_chain_length
        self.max_chain_length = max_chain_length
        self.include_meta = include_meta
        self.sort_by = sort_by or ["-frequency", "-length"]
        self.system_prompt = system_prompt
        self.user_instruction = user_instruction
        self.num_workers = int(num_workers)

        self._data: list[Dict[str, Any]] = []

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = ReasonerPathDataset(
            eval_cache_path=str(self.eval_cache_path),
            entity_vocab_path=str(self.entity_vocab_path),
            relation_vocab_path=str(self.relation_vocab_path) if self.relation_vocab_path else None,
            g_agent_path=str(self.g_agent_path) if self.g_agent_path else None,
            questions_path=str(self.questions_path),
            answer_text_field=self.answer_text_field,
            artifact_name=self.artifact_name,
            schema_version=self.schema_version,
            max_chains_per_sample=self.max_chains_per_sample,
            min_chain_length=self.min_chain_length,
            max_chain_length=self.max_chain_length,
            include_meta=self.include_meta,
            sort_by=self.sort_by,
            system_prompt=self.system_prompt,
            user_instruction=self.user_instruction,
            prompt_tag=self.prompt_tag,
        )
        # Materialize into list to simplify collate (and keep deterministic ordering)
        self._data = [dataset[i] for i in range(len(dataset))]

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            _ListDataset(self._data),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )


__all__ = ["ReasonerPathDataModule"]
