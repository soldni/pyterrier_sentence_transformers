from functools import cached_property
from typing import cast
import numpy as np

import pandas as pd
from pyterrier.model import add_ranks

from .base import SentenceTransformersBase


class SentenceTransformersRetriever(SentenceTransformersBase):
    @cached_property
    def faiss_index(self):
        index = super().faiss_index
        index.deserialize_from(self.config.faiss_index_path)
        return index

    @cached_property
    def num_results(self) -> int:
        return min(self.config.num_results, self.faiss_index.index.ntotal)

    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:

        queries_embeddings = self.model.encode(
            topics[self.config.query_attr].to_list(),
            batch_size=self.config.per_gpu_eval_batch_size,
            show_progress_bar=self.config.verbose,
            convert_to_tensor=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        queries_embeddings = cast(np.ndarray, queries_embeddings)
        results = self.faiss_index.search_knn(
            query_vectors=queries_embeddings,
            top_docs=self.num_results
        )

        queries_accumulator = []
        for i, (doc_ids, scores) in enumerate(results):
            query_results = pd.DataFrame({
                'qid': [topics['qid'].iloc[i] for _ in doc_ids],
                'docno': doc_ids,
                'score': scores,
            })
            query_results = add_ranks(query_results)
            queries_accumulator.append(query_results)

        all_results = pd.concat(queries_accumulator)
        all_results = all_results.sort_values(
            by=["qid", "score", "docno"], ascending=[True, False, True]
        )
        return all_results
