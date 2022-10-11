from functools import cached_property
import pyterrier as pt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union, cast
from pyterrier.transformer import TransformerBase
from sentence_transformers import SentenceTransformer

import torch


@dataclass
class SentenceTransformerConfig:
    model_name_or_path: str
    index_path: str
    verbose: bool = False
    num_docs: Optional[int] = None
    local_rank: int = -1
    cache_dir: Optional[str] = None
    text_attr: str = "text"
    docno_attr: str = "docno"
    query_attr: str = "query"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_query_length: Optional[int] = None
    max_doc_length: Optional[int] = None
    per_gpu_eval_batch_size: int = 128
    per_call_size: int = 1_024
    num_results: int = 1000
    faiss_n_subquantizers: int = 0
    faiss_n_bits: int = 8
    n_gpu: int = torch.cuda.device_count()

    @property
    def faiss_index_path(self) -> str:
        path = Path(self.index_path, 'faiss')
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @classmethod
    def from_dict(
        cls: Type['SentenceTransformerConfig'],
        config: Optional[Union['SentenceTransformerConfig', dict]] = None,
        override_dict: Optional[Dict[str, Any]] = None
    ) -> 'SentenceTransformerConfig':

        if not (
            config is None or
            isinstance(config, (dict, cls))
        ):
            raise TypeError(
                f"config must be a dict or {cls.__name__}, "
                f"not {type(config)}"
            )

        if not (override_dict is None or isinstance(override_dict, dict)):
            raise TypeError(
                f"override_dict must be a dict, not {type(override_dict)}"
            )

        if isinstance(config, cls):
            config = asdict(config)
        elif config is None:
            config = {}

        config_with_overrides = {**cast(dict, config), **(override_dict or {})}

        return cls(**config_with_overrides)


class SentenceTransformersBase(TransformerBase):
    def __init__(
        self,
        config: Optional[Union[SentenceTransformerConfig, dict]] = None,
        **kwargs
    ):
        super().__init__()
        self.config = SentenceTransformerConfig.from_dict(config, kwargs)

    def __getstate__(self) -> dict:
        """Return the state of the object for pickling, minus
        any cached properties value, which might not be serializable
        and/or are better to recompute on load in case of multiprocessing."""

        state = self.__dict__.copy()

        # do not serialize the result of cached properties
        for k in list(state.keys()):
            if isinstance(getattr(type(self), k, None), cached_property):
                del state[k]

        return state

    def __setstate__(self, state: dict):
        """Restore the state of the object after pickling.
        Cached properties will be recomputed on first access."""
        self.__dict__.update(state)

    @cached_property
    def model(self) -> SentenceTransformer:
        model = SentenceTransformer(
            model_name_or_path=self.config.model_name_or_path,
            device=self.config.device,
        )
        if self.config.max_doc_length:
            model.max_seq_length = self.config.max_doc_length
        return model

    def get_pbar(self, obj_to_iterate: Any, desc: str, unit: str) -> pt.tqdm:
        """Get a progress bar for the given object to iterate over."""

        if self.config.num_docs is not None:
            total = self.config.num_docs
        elif hasattr(obj_to_iterate, "__len__"):
            total = len(obj_to_iterate)
        else:
            total = None

        return pt.tqdm(
            obj_to_iterate,
            total=total,
            desc=desc,
            unit=f" {unit.strip()}",
            unit_scale=True,
            disable=not self.config.verbose
        )
