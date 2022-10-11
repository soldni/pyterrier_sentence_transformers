import unittest
import tempfile


class TestIndexing(unittest.TestCase):
    def test_indexing_vaswani(self):

        from pyterrier.datasets import get_dataset
        from pyterrier_sentence_transformers import (
            SentenceTransformersIndexer,
            SentenceTransformersRetriever
        )

        self.tearDown()
        self.setUp()

        indexer = SentenceTransformersIndexer(
            model_name_or_path="all-MiniLM-L6-v2",
            index_path=self.test_dir
        )

        corpus_iter = get_dataset("vaswani").get_corpus_iter()
        ref = indexer.index([next(corpus_iter) for i in range(200)])
        ret = SentenceTransformersRetriever(
            model_name_or_path="all-MiniLM-L6-v2",
            index_path=ref
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
        except Exception:
            pass
