# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Originally imported from https://github.com/facebookresearch/contriever/blob/27b5ebb476df1b31b05910980f05277a02f689c2/src/index.py  # noqa: E501

Author: Luca Soldaini
GitHub: @soldni
Email:  lucas@allenai.org
'''

from enum import Enum
from multiprocessing import cpu_count
import os
import pickle
from typing import Any, List, Tuple, Union

from necessary import necessary
import numpy as np
from tqdm import tqdm

with necessary(
    modules="faiss",
    message=(
        "Faiss is not installed. Installation instructions can be found at "
        "https://github.com/facebookresearch/faiss/blob/main/INSTALL.md; if "
        "on macOS with Apple Silicon, you can use the script at "
        "scripts/as_install_faiss.sh."
    )
):
    import faiss


class FaissMetric(Enum):
    METRIC_INNER_PRODUCT = faiss.METRIC_INNER_PRODUCT   # pyright: ignore
    METRIC_L2 = faiss.METRIC_L2                         # pyright: ignore
    METRIC_L1 = faiss.METRIC_L1                         # pyright: ignore
    METRIC_Linf = faiss.METRIC_Linf                     # pyright: ignore
    METRIC_Lp = faiss.METRIC_Lp                         # pyright: ignore
    METRIC_Canberra = faiss.METRIC_Canberra             # pyright: ignore
    METRIC_BrayCurtis = faiss.METRIC_BrayCurtis         # pyright: ignore
    METRIC_JensenShannon = faiss.METRIC_JensenShannon   # pyright: ignore


class FaissIndex(object):
    def __init__(
        self,
        vector_sz: int,
        factory_config: str = 'Flat',
        factory_metric: Union[FaissMetric, str] = 'METRIC_INNER_PRODUCT',
        n_threads: int = cpu_count() // 2,
    ):
        if isinstance(factory_metric, str):
            factory_metric = FaissMetric[factory_metric]

        faiss.omp_set_num_threads(n_threads)    # pyright: ignore

        self.index = faiss.index_factory(       # pyright: ignore
            vector_sz, factory_config, factory_metric.value
        )

        self.index_id_to_db_id: List[Any] = []

    def index_data(self, ids: List[Any], embeddings: np.ndarray):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')

        if not self.index.is_trained:
            self.index.train(embeddings)

        self.index.add(embeddings)

        print(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(
        self,
        query_vectors: np.ndarray,
        top_docs: int,
        index_batch_size: int = 2048
    ) -> List[Tuple[List[Any], List[Any]]]:
        query_vectors = query_vectors.astype('float32')
        result = []
        n_batch = (len(query_vectors) - 1) // index_batch_size + 1

        for k in tqdm(range(n_batch)):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [
                [str(self.index_id_to_db_id[i]) for i in query_top_idxs]
                for query_top_idxs in indexes
            ]

            result.extend([
                (db_ids[i], scores[i]) for i in range(len(db_ids))
            ])

        return result

    def serialize(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Serializing index to {index_file}, meta data to {meta_file}')

        faiss.write_index(self.index, index_file)   # pyright: ignore
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)   # pyright: ignore
        print(
            'Loaded index of type %s and size %d', type(self.index),
            self.index.ntotal
        )

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(self.index_id_to_db_id) == self.index.ntotal, (
            'Deserialized index_id_to_db_id should match faiss index size'
        )

    def _update_id_mapping(self, db_ids: List):
        # new_ids = np.array(db_ids, dtype=np.int64)
        # self.index_id_to_db_id = np.concatenate(
        #   (self.index_id_to_db_id, new_ids),
        # axis=0)
        self.index_id_to_db_id.extend(db_ids)
