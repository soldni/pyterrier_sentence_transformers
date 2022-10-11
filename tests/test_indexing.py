import unittest
import pandas as pd
import tempfile


class TestIndexing(unittest.TestCase):
    def test_indexing_one_doc(self):
        import pyterrier as pt
        from pyterrier_sentence_transformers import SentenceTransformersIndexer

        self.tearDown()
        self.setUp()

        indexer = SentenceTransformersIndexer(
            model_name_or_path="all-MiniLM-L6-v2",
            index_path=self.test_dir
        )
        df = pd.DataFrame({"docno": [1], "text": ["This is a test"]})
        indexref = indexer.index(df)
        index = pt.IndexFactory.of(indexref)
        assert index.getCollectionStatistics().getNumberOfDocuments() == 1

    def test_indexing_vaswani(self):

        from pyterrier.datasets import get_dataset
        from pyterrier_sentence_transformers import (
            SentenceTransformersIndexer,
            SentenceTransformersRetrieval
        )

        self.tearDown()
        self.setUp()

        indexer = SentenceTransformersIndexer(
            model_name_or_path="all-MiniLM-L6-v2",
            index_path=self.test_dir
        )

        iter = get_dataset("vaswani").get_corpus_iter()
        ref = indexer.index([ next(iter) for i in range(200) ])
        ret = SentenceTransformersRetrieval(
            model_name_or_path="all-MiniLM-L6-v2",
            indexref=ref
        )

        dfOut = ret.search("chemical reactions")
        self.assertTrue(len(dfOut) > 0)

    def setUp(self):
        import pyterrier as pt
        if not pt.started():
            pt.init()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass
