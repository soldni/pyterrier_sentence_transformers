from pathlib import Path
import shutil
from typing import Optional
import platformdirs
import pyterrier as pt

if not pt.started():
    pt.init()

from pyterrier_sentence_transformers import (
    SentenceTransformersRetriever,
    SentenceTransformersIndexer,
)

import pandas as pd


DATASET_NAME = 'from_json'
NEU_MODEL_NAME = 'facebook/contriever-msmarco'
STA_MODEL_NAME = 'BM25'


class StripMarkup():
    # following https://github.com/terrier-org/pyterrier/issues/253

    def __init__(self):
        self.tokenizer = pt.autoclass(
            "org.terrier.indexing.tokenisation.Tokeniser"
        ).getTokeniser()

    def __call__(self, text):
        return " ".join(self.tokenizer.getTokens(text))


def main():
    # fixed keys: docno, qid, doc_id, query, relevance

    corpus = [
        {'docno': '123', 'text': 'The Golems Eye'},
        {'docno': '124', 'text': 'Weslandia'},
        {'docno': '125', 'text': 'Ship Breaker'},
        {'docno': '126', 'text': 'Dreams of Kim Edwards'},
        {'docno': '127', 'text': 'The Lake of a Big Dream'},
        {'docno': '128', 'text': 'The Lavender Shoes'},
        {'docno': '129', 'text': 'The Lavender House'},
    ]

    topics = pd.DataFrame.from_records([
        {'qid': 'en32fo', 'query': 'The Lake of Dreams by Kim Edwards'},
        {'qid': 'j1doae', 'query': "The Lavender House: A Dystopian Children's Story"}
    ])

    qrels = pd.DataFrame.from_records([
        {'qid': 'en32fo', 'doc_id': '127', 'relevance': 1},
        {'qid': 'j1doae', 'doc_id': '129', 'relevance': 1}
    ])

    index_root = Path(
        platformdirs.user_cache_dir('pyterrier_sentence_transformers')
    ) / DATASET_NAME

    if index_root.exists():
        shutil.rmtree(index_root)

    # This is the neural indexer with sentence-transformers
    neu_index_path = index_root / NEU_MODEL_NAME.replace('/', '_')
    neu_index_path.mkdir(parents=True, exist_ok=True)
    indexer = SentenceTransformersIndexer(
        model_name_or_path=NEU_MODEL_NAME,
        index_path=str(neu_index_path),
        overwrite=True,
        normalize=False,
        text_attr=['text']
    )
    indexer.index(corpus)

    # This is a classic statistical indexer
    sta_index_path = index_root / STA_MODEL_NAME
    sta_index_path.mkdir(parents=True, exist_ok=True)
    if not (sta_index_path / 'data.properties').exists():
        indexer = pt.IterDictIndexer(
            index_path=str(sta_index_path),
            blocks=True,
            tokeniser="UTFTokeniser"
        )
        indexref = indexer.index(corpus, fields=['text'])
        index = pt.IndexFactory.of(indexref)
    else:
        index = pt.IndexFactory.of(str(sta_index_path))


    # Retrievers (neural and statistical)
    neu_retr = SentenceTransformersRetriever(
        model_name_or_path=NEU_MODEL_NAME,
        index_path=str(neu_index_path)
    )

    sta_retr = pt.BatchRetrieve(index, wmodel=STA_MODEL_NAME)

    markup_stripper = StripMarkup()
    topics = pt.apply.query(lambda r: markup_stripper(r.query))(topics)

    # run the experiment
    exp = pt.Experiment(
        [sta_retr, neu_retr],
        topics,
        qrels,
        names=[STA_MODEL_NAME, NEU_MODEL_NAME],
        eval_metrics=["map", "P.1"]
    )
    print(exp)

    # to search, use stat_retr.search(string) or neu_retr.search(string)


if __name__ == '__main__':
    main()
