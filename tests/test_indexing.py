import unittest
import pandas as pd
import tempfile
class TestIndexing(unittest.TestCase):

    def test_indexing_1doc_torch(self):
        import pyterrier as pt
        from pyterrier_ance import ANCEIndexer, ANCERetrieval
        checkpoint="https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip"
        #checkpoint="/Users/craigmacdonald/git/pyterrier_ance/ance_checkpoint.zip"
        import os
        os.rmdir(self.test_dir)
        indexer = ANCEIndexer(
            checkpoint, 
            os.path.join(self.test_dir, "index"), 
            num_docs=200
            )

        iter = pt.get_dataset("vaswani").get_corpus_iter()
        ref = indexer.index([ next(iter) for i in range(200) ])
        ret = ANCERetrieval(checkpoint, ref)
       
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