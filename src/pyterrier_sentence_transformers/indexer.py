from functools import cached_property
from pathlib import Path
import shutil
from typing import Any, Iterable
import more_itertools

import pandas as pd
from trouting import trouting

from .base import SentenceTransformersBase

from pyterrier.datasets import GeneratorLen

GeneratorType = type((i for i in range(10)))


class SentenceTransformersIndexer(SentenceTransformersBase):

    @trouting
    def make_segments(self, docs: Any) -> Iterable[pd.DataFrame]:
        raise NotImplementedError(
            f"Cannot make segments from object of type {type(docs)}."
        )

    @make_segments.add_interface(docs=(list, GeneratorType, GeneratorLen))
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

    @cached_property
    def sep_token(self) -> str:
        sep_token = getattr(self.model.tokenizer, "sep_token")
        if sep_token is None:
            raise ValueError('No sep_token found in tokenizer')
        return str(sep_token)

    def index(self, docs):
        # make sure input path exists
        path = Path(self.config.index_path)
        if not path.exists() or self.config.overwrite:
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)

        # get a nice progress bar to show user how indexing is going
        pbar = self.get_pbar(docs, desc="Indexing", unit="docs")

        # we index in segments (i.e. batches) to avoid improve throughput
        # and maybe in the future to shard the index across multiple files
        docs_it = self.make_segments(docs)

        for _, contents in enumerate(docs_it):

            # this is all the text fields we want to index
            text_contents = contents[self.config.text_attr].apply(
                # join using sep_token
                lambda row: f' {self.sep_token} '.join(row.values.astype(str)),
                # join along columns
                axis=1
            ).to_list()

            passage_embedding = self.encode(texts=text_contents)

            self.faiss_index.index_data(
                ids=[str(e) for e in contents[self.config.docno_attr]],
                embeddings=passage_embedding
            )
            # update progress bar
            pbar.update(passage_embedding.shape[0])

        self.faiss_index.serialize(self.config.faiss_index_path)
        return self.config.index_path
