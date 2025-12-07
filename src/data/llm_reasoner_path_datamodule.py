from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .llm_reasoner_path_dataset import LLMReasonerPathDataset


class _ListDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


class LLMReasonerPathDataModule(LightningDataModule):
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
        dataset: str = "webqsp",
        split: str = "test",
        prompt_tag: str = "paths",
        max_chains_per_sample: int = 10,
        min_chain_length: int = 1,
        max_chain_length: Optional[int] = None,
        include_meta: bool = True,
        sort_by: Optional[list[str]] = None,
        system_prompt: str = "You are a concise QA agent. Use the provided paths as evidence.",
        user_instruction: str = (
            "Copy the answer entity exactly as it appears in the paths (no paraphrase or alias). "
            "Prefix with 'Ans:'. If the answer is absent, reply 'Ans: Unknown'."
        ),
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.eval_cache_path = Path(eval_cache_path).expanduser().resolve()
        self.entity_vocab_path = Path(entity_vocab_path).expanduser().resolve()
        self.relation_vocab_path = Path(relation_vocab_path).expanduser().resolve() if relation_vocab_path else None
        self.dataset = dataset
        self.split = split
        self.prompt_tag = prompt_tag
        self.max_chains_per_sample = max_chains_per_sample
        self.min_chain_length = min_chain_length
        self.max_chain_length = max_chain_length
        self.include_meta = include_meta
        self.sort_by = sort_by or ["-frequency", "-success_hits", "-length"]
        self.system_prompt = system_prompt
        self.user_instruction = user_instruction
        self.num_workers = int(num_workers)

        self._data: list[Dict[str, Any]] = []

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = LLMReasonerPathDataset(
            eval_cache_path=str(self.eval_cache_path),
            entity_vocab_path=str(self.entity_vocab_path),
            relation_vocab_path=str(self.relation_vocab_path) if self.relation_vocab_path else None,
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


__all__ = ["LLMReasonerPathDataModule"]
