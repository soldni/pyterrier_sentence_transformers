from functools import cached_property
import hashlib
import json
import numpy as np
import pyterrier as pt
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, cast
from pyterrier.transformer import TransformerBase
from sentence_transformers import SentenceTransformer

import torch
from .index import FaissIndex


@dataclass
class SentenceTransformerConfig:
    model_name_or_path: str
    index_path: str
    overwrite: bool = False
    verbose: bool = False
    num_docs: Optional[int] = None
    local_rank: int = -1
    cache_dir: Optional[str] = None
    text_attr: List[str] = field(default_factory=lambda: ["text"])
    docno_attr: str = "docno"
    query_attr: str = "query"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_length: Optional[int] = None
    per_gpu_eval_batch_size: int = 128
    per_call_size: int = 1_024
    num_results: int = 1000
    normalize: bool = True
    faiss_factory_config: str = 'Flat'
    faiss_factory_metric: str = 'METRIC_INNER_PRODUCT'
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

    def __str__(self):
        h = hashlib.sha1()
        h.update(
            json.dumps(asdict(self.config), sort_keys=True).encode('utf-8')
        )
        return (
            f"{self.__class__.__name__}"
            f"({self.config.model_name_or_path},{h.hexdigest()[:8]})"
        )

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
        if self.config.max_length:
            model.max_seq_length = self.config.max_length
        return model

    @cached_property
    def faiss_index(self) -> FaissIndex:

        embedding_size = self.model.get_sentence_embedding_dimension()
        assert isinstance(embedding_size, int), (
            f"get_sentence_embedding_dimension returned {embedding_size}, "
            f"which is of type {type(embedding_size)}, not int."
        )

        # make the index here; it will be index in memory first, and
        # then written to disk
        index = FaissIndex(
            vector_sz=embedding_size,
            factory_config=self.config.faiss_factory_config,
            factory_metric=self.config.faiss_factory_metric,
        )

        return index

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode the given texts using the model."""

        out = self.model.encode(
            texts,
            batch_size=self.config.per_gpu_eval_batch_size,
            convert_to_tensor=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize
        )
        return cast(np.ndarray, out)

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
