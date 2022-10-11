from pathlib import Path
from typing import Any, Iterable, cast
import more_itertools
import numpy as np

import pandas as pd

from trouting import trouting

from .index import FaissIndex
from .base import SentenceTransformersBase


GeneratorType = type((i for i in range(10)))


class SentenceTransformersIndexer(SentenceTransformersBase):

    @trouting
    def make_segments(self, docs: Any) -> Iterable[pd.DataFrame]:
        raise NotImplementedError(
            f"Cannot make segments from object of type {type(docs)}."
        )

    @make_segments.add_interface(docs=(list, GeneratorType))
    def _make_segments_from_iterable(
        self, docs: Iterable[dict]
    ) -> Iterable[pd.DataFrame]:
        for shard in more_itertools.chunked(docs, self.config.per_call_size):
            yield pd.DataFrame.from_records(shard)

    @make_segments.add_interface(docs=pd.DataFrame)
    def _make_segments_from_dataframe(
        self, docs: pd.DataFrame
    ) -> Iterable[pd.DataFrame]:
        chunk_size = self.config.per_call_size
        num_chunks = (len(docs) // chunk_size) + 1
        for i in range(num_chunks):
            yield docs[i * chunk_size : (i+1) * chunk_size]

    def index(self, docs):
        # make sure input path exists
        Path(self.config.index_path).mkdir(parents=True, exist_ok=True)

        # get a nice progress bar to show user how indexing is going
        pbar = self.get_pbar(docs, desc="Indexing", unit="docs")

        # we index in segments (i.e. batches) to avoid improve throughput
        # and maybe in the future to shard the index across multiple files
        docs_it = self.make_segments(docs)

        embedding_size = self.model.get_sentence_embedding_dimension()
        assert isinstance(embedding_size, int), (
            f"get_sentence_embedding_dimension returned {embedding_size}, "
            f"which is of type {type(embedding_size)}, not int."
        )

        # make the index here; it will be index in memory first, and
        # then written to disk
        index = FaissIndex(
            vector_sz=embedding_size,
            n_bits=self.config.faiss_n_bits,
            n_subquantizers=self.config.faiss_n_subquantizers,
        )

        for _, contents in enumerate(docs_it):
            passage_embedding = self.model.encode(
                contents[self.config.text_attr].to_list(),
                batch_size=self.config.per_gpu_eval_batch_size,
                show_progress_bar=self.config.verbose,
                convert_to_tensor=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            passage_embedding = cast(np.ndarray, passage_embedding)

            index.index_data(
                ids=[str(e) for e in contents[self.config.docno_attr]],
                embeddings=passage_embedding
            )
            # update progress bar
            pbar.update(passage_embedding.shape[0])

        index.serialize(self.config.faiss_index_path)
        return self.config.index_path
