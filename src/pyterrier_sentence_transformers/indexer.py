import os
import pickle
from dataclasses import asdict, dataclass
from typing import Generator, Iterable, Optional, Union

import numpy as np
import pandas as pd
import pyterrier as pt
import torch
from pyterrier.transformer import TransformerBase
from sentence_transformers import SentenceTransformer

from necessary import necessary

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


@dataclass
class SentenceTransformerIndexerConfig:
    model_name_or_path: str
    index_path: str
    verbose: bool = False
    num_docs: Optional[int] = None
    local_rank: int = -1
    cache_dir: Optional[str] = None
    text_attr: str = "text"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_query_length: Optional[int] = None
    max_doc_length: Optional[int] = None
    per_gpu_eval_batch_size: int = 128
    n_gpu: int = torch.cuda.device_count()

    @property
    def doc_tokenizer_kwargs(self):
        return dict(
            max_length=self.max_doc_length,
            truncation=True,
        )

    @property
    def doc_query_tokenizer_kwargs(self):
        return dict(
            max_length=self.max_query_length,
            truncation=True,
        )


class SentenceTransformersIndexer(TransformerBase):

    def __init__(
        self,
        config: Optional[Union[SentenceTransformerIndexerConfig, dict]] = None,
        **kwargs
    ):
        if isinstance(config, SentenceTransformerIndexerConfig):
            config = asdict(config)
        elif config is None:
            config = {}
        config.update(kwargs)
        self.config = SentenceTransformerIndexerConfig(**config)

    @classmethod
    def load_model(
        cls,
        config: SentenceTransformerIndexerConfig
    ) -> SentenceTransformer:
        return SentenceTransformer(
            model_name_or_path=config.model_name_or_path,
            device=config.device
        )

    def index(self, docs_it: Iterable[pd.DataFrame]):
        # from ance.utils.util import pad_input_ids
        # import torch
        # import more_itertools
        # import pyterrier as pt
        # from ance.drivers.run_ann_data_gen import StreamInferenceDoc, load_model, GetProcessingFn
        # import ance.drivers.run_ann_data_gen
        # import pickle
        # import os
        # # monkey patch ANCE to use the same TQDM as PyTerrier
        # ance.drivers.run_ann_data_gen.tqdm = pt.tqdm

        # import os
        # os.makedirs(self.index_path)

        docid2docno = []

        # def gen_tokenize():
        #     text_attr = self.text_attr
        #     kwargs = {}
        #     if self.num_docs is not None:
        #         kwargs['total'] = self.num_docs
        #     for doc in pt.tqdm(generator, desc="Indexing", unit="d", **kwargs) if self.verbose else generator:
        #         contents = doc[text_attr]
        #         docid2docno.append(doc["docno"])

        #         passage = tokenizer.encode(
        #             contents,
        #             add_special_tokens=True,
        #             max_length=self.args.max_seq_length,
        #         )
        #         passage_len = min(len(passage), self.args.max_seq_length)
        #         input_id_b = pad_input_ids(passage, self.args.max_seq_length)
        #         yield passage_len, input_id_b

        segment = -1
        shard_size = []

        if self.config.verbose:
            tqdm_kwargs = {}
            if self.config.num_docs is not None:
                tqdm_kwargs['total'] = self.config.num_docs

            docs_it = pt.tqdm(
                docs_it, desc="Indexing", unit="d", **tqdm_kwargs
            )

        for doc in docs_it:
            contents = doc[self.config.text_attr]

            docid2docno.append(doc["docno"])
            segment += 1

            print("Segment %d" % segment)

            passage_embedding, passage_embedding2id = StreamInferenceDoc(self.args, model, GetProcessingFn(
                self.args, query=False), "passages", gengen, is_query_inference=False)

            dim=passage_embedding.shape[1]
            faiss.omp_set_num_threads(16)
            cpu_index = faiss.IndexFlatIP(dim)
            cpu_index.add(passage_embedding)
            faiss_file = os.path.join(self.index_path, str(segment) + ".faiss")
            lookup_file = os.path.join(self.index_path, str(segment) + ".docids.pkl")

            faiss.write_index(cpu_index, faiss_file)
            cpu_index = None
            passage_embedding = None

            with pt.io.autoopen(lookup_file, 'wb') as f:
                pickle.dump(passage_embedding2id, f)
            shard_size.append(len(passage_embedding2id))
            passage_embedding2id = None

        with pt.io.autoopen(os.path.join(self.index_path, "shards.pkl"), 'wb') as f:
            pickle.dump(shard_size, f)
            pickle.dump(docid2docno, f)
        return self.index_path
