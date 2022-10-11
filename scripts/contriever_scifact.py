from pathlib import Path
from typing import Optional
import platformdirs
import pyterrier as pt

if not pt.started():
    pt.init()

from pyterrier_sentence_transformers import (
    SentenceTransformersRetriever,
    SentenceTransformersIndexer,
    SentenceTransformerConfig
)

import springs as sp

DATASET = 'beir/scifact/test'
NEU_MODEL_NAME = 'facebook/contriever-msmarco'
STA_MODEL_NAME = 'BM25'


@sp.dataclass
class SentenceTransformerConfigWithDefaults(SentenceTransformerConfig):
    model_name_or_path: Optional[str] = None    # type: ignore
    index_path: Optional[str] = None            # type: ignore


@sp.cli(SentenceTransformerConfigWithDefaults)
def main(config: SentenceTransformerConfigWithDefaults):
    dataset = pt.get_dataset(f'irds:{DATASET}')
    index_root = Path(
        platformdirs.user_cache_dir('pyterrier_sentence_transformers')
    ) / DATASET.replace('/', '_')

    # This is the neural indexer with sentence-transformers
    neu_index_path = index_root / NEU_MODEL_NAME.replace('/', '_')
    neu_index_path.mkdir(parents=True, exist_ok=True)
    indexer = SentenceTransformersIndexer(
        model_name_or_path=NEU_MODEL_NAME,
        index_path=str(neu_index_path),
        overwrite=True,
        normalize=False,
        config=sp.to_dict(config)
    )
    indexer.index(dataset.get_corpus_iter())

    # This is a classic statistical indexer
    sta_index_path = index_root / STA_MODEL_NAME
    if not (sta_index_path / 'data.properties').exists():
        indexer = pt.IterDictIndexer(
            index_path=str(sta_index_path), blocks=True
        )
        indexref = indexer.index(dataset.get_corpus_iter())
        index = pt.IndexFactory.of(indexref)
    else:
        index = pt.IndexFactory.of(str(sta_index_path))

    # Retrievers (neural and statistical)
    neu_retr = SentenceTransformersRetriever(
        model_name_or_path=NEU_MODEL_NAME,
        index_path=str(neu_index_path)
    )
    sta_retr = pt.BatchRetrieve(index, wmodel=STA_MODEL_NAME)

    # run the experiement
    exp = pt.Experiment(
        [sta_retr, neu_retr],
        dataset.get_topics(),
        dataset.get_qrels(),
        names=[STA_MODEL_NAME, NEU_MODEL_NAME],
        eval_metrics=["map", "recip_rank", "P.10", "ndcg_cut.10"]
    )
    print(exp)

if __name__ == '__main__':
    main()
