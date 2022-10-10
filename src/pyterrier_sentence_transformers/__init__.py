from pyterrier.datasets import Dataset
from typing import Union
from pyterrier.transformer import TransformerBase


def _load_model(args, checkpoint_path):
    from ance.drivers.run_ann_data_gen import load_model
    # support downloads of checkpoints
    if checkpoint_path.startswith("http"):
        print("Downloading checkpoint %s" % checkpoint_path)
        import tempfile, wget
        targetZip = os.path.join(tempfile.mkdtemp(), 'checkpoint.zip')
        wget.download(checkpoint_path, targetZip)
        checkpoint_path = targetZip
    
    # support zip files of checkpoints
    if checkpoint_path.endswith(".zip"):
        import tempfile, zipfile
        print("Extracting checkpoint %s" % checkpoint_path)
        targetDir = tempfile.mkdtemp()
        zipfile.ZipFile(checkpoint_path).extractall(targetDir)
        #todo fix this
        checkpoint_path = os.path.join(targetDir, "Passage ANCE(FirstP) Checkpoint")

    print("Loading checkpoint %s" % checkpoint_path)
    config, tokenizer, model = load_model(args, checkpoint_path)
    return config, tokenizer, model


class ANCEIndexer(TransformerBase):
    
    def __init__(self, checkpoint_path, index_path, num_docs=None, verbose=True, text_attr="text", segment_size=500_000, **kwargs):
        self.index_path = index_path
        self.checkpoint_path = checkpoint_path
        self.verbose=verbose
        self.num_docs = num_docs
        if self.verbose and self.num_docs is None:
            raise ValueError("if verbose=True, num_docs must be set")
        self.segment_size = segment_size
        self.text_attr = text_attr

        args = type('', (), {})()
        args.local_rank = -1
        args.model_type = 'rdot_nll'
        args.cache_dir  = None
        args.no_cuda = False
        args.max_query_length = 64
        args.max_seq_length = 128

        args.per_gpu_eval_batch_size = 128
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.__dict__.update(kwargs)
        self.args = args
        
    def index(self, generator):
        from ance.utils.util import pad_input_ids
        import torch
        import more_itertools
        import pyterrier as pt
        from ance.drivers.run_ann_data_gen import StreamInferenceDoc, load_model, GetProcessingFn
        import ance.drivers.run_ann_data_gen
        import pickle
        import os
        # monkey patch ANCE to use the same TQDM as PyTerrier
        ance.drivers.run_ann_data_gen.tqdm = pt.tqdm

        import os
        os.makedirs(self.index_path)

        config, tokenizer, model = _load_model(self.args, self.checkpoint_path)

        
        docid2docno = []
        def gen_tokenize():
            text_attr = self.text_attr
            kwargs = {}
            if self.num_docs is not None:
                kwargs['total'] = self.num_docs
            for doc in pt.tqdm(generator, desc="Indexing", unit="d", **kwargs) if self.verbose else generator:
                contents = doc[text_attr]
                docid2docno.append(doc["docno"])
                
                passage = tokenizer.encode(
                    contents,
                    add_special_tokens=True,
                    max_length=self.args.max_seq_length,
                )
                passage_len = min(len(passage), self.args.max_seq_length)
                input_id_b = pad_input_ids(passage, self.args.max_seq_length)
                yield passage_len, input_id_b
        
        segment=-1
        shard_size=[]
        for gengen in more_itertools.ichunked(gen_tokenize(), self.segment_size):
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

import faiss
from pyterrier.model import add_ranks
from ance.drivers.run_ann_data_gen import StreamInferenceDoc, load_model, GetProcessingFn
import pickle
import torch
from ance.utils.util import pad_input_ids
import os
import pyterrier as pt
import pandas as pd
import numpy as np

class ANCERetrieval(TransformerBase):

    def __init__(self, checkpoint_path=None, index_path=None, cpu_index=None, passage_embedding2id = None, docid2docno=None, num_results=100, **kwargs):
        self.args = type('', (), {})()
        args = self.args
        args.local_rank = -1
        args.model_type = 'rdot_nll'
        args.cache_dir  = None
        args.no_cuda = False
        args.max_query_length = 64
        args.max_seq_length = 128
        args.per_gpu_eval_batch_size = 128
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.__dict__.update(kwargs)
        
        self.checkpoint_path = checkpoint_path
        self.num_results = num_results
        from pyterrier import tqdm

        #faiss.omp_set_num_threads(16)
        
        config, tokenizer, model = _load_model(self.args, self.checkpoint_path)
        self.model = model
        self.tokenizer = tokenizer
        if index_path is not None:
            print("Loading shard metadata")
            shards_files = os.path.join(index_path, "shards.pkl")
            with pt.io.autoopen(shards_files) as f:
                self.shard_sizes = pickle.load(f)
                self.docid2docno = pickle.load(f)
            self.segments = len(self.shard_sizes)
            self.cpu_index = []
            self.shard_offsets = []
            self.passage_embedding2id = []
            offset=0
            for i, shard_size in enumerate(tqdm(self.shard_sizes, desc="Loading shards", unit="shard")):                
                faiss_file = os.path.join(index_path, str(i) + ".faiss")
                lookup_file = os.path.join(index_path, str(i) + ".docids.pkl")
                index = faiss.read_index(faiss_file)
                self.cpu_index.append(index)
                self.shard_offsets.append(offset)
                offset += shard_size
                with pt.io.autoopen(lookup_file) as f:
                    self.passage_embedding2id.append(pickle.load(f))
        else:
            self.cpu_index = cpu_index
            self.passage_embedding2id = passage_embedding2id
            self.docid2docno = docid2docno

    #allows a colbert ranker to be built from a dataset
    def from_dataset(dataset : Union[str,Dataset], 
            variant : str = None, 
            version='latest',            
            **kwargs):

        from pyterrier.batchretrieve import _from_dataset

        #ANCERetrieval doesnt match quite the expectations, so we can use a wrapper fn
        def _ANCERetrievalconstruct(folder, **kwargs):
            import os
            checkpoint_path = kwargs.get('checkpoint_path')
            del kwargs['checkpoint_path']
            return ANCERetrieval(checkpoint_path, folder, **kwargs)

        return _from_dataset(dataset, 
                             variant=variant, 
                             version=version, 
                             clz=_ANCERetrievalconstruct, **kwargs)

    def __str__(self):
        return "ANCE"

    def transform(self, topics):
        from pyterrier import tqdm
        queries=[]
        qid2q = {}
        for q, qid in zip(topics["query"].to_list(), topics["qid"].to_list()):
            passage = self.tokenizer.encode(
                q,
                add_special_tokens=True,
                max_length=self.args.max_seq_length,
            )
                
            passage_len = min(len(passage), self.args.max_query_length)
            input_id_b = pad_input_ids(passage, self.args.max_query_length)
            queries.append([passage_len, input_id_b])
            qid2q[qid] = q
        
        print("***** inference of %d queries *****" % len(queries))
        dev_query_embedding, dev_query_embedding2id = StreamInferenceDoc(self.args, self.model, GetProcessingFn(
             self.args, query=True), "transform", queries, is_query_inference=True)
        
        
        print("***** faiss search for %d queries on %d shards *****" % (len(queries), self.segments))
        rtr = []
        for i, offset in enumerate(tqdm(self.shard_offsets, unit="shard")):
            scores, neighbours = self.cpu_index[i].search(dev_query_embedding, self.num_results)
            res = self._calc_scores(topics["qid"].values, self.passage_embedding2id[i], neighbours, scores, num_results=self.num_results, offset=offset, qid2q=qid2q)
            rtr.append(res)
        rtr = pd.concat(rtr)
        rtr = add_ranks(rtr)
        rtr = rtr[rtr["rank"] < self.num_results]
        rtr = rtr.sort_values(by=["qid", "score", "docno"], ascending=[True, False, True])
        return rtr

    def _calc_scores(self, 
        query_embedding2id,
        passage_embedding2id,
        I_nearest_neighbor, I_scores, num_results=50, offset=0, qid2q=None):
        """
            based on drivers.run_ann_data_gen.EvalDevQuery
        """
        # NB The Microsof impl used -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)
        # We use the scores from Faiss
        rtr=[]
        for query_idx in range(I_nearest_neighbor.shape[0]):
            query_id = query_embedding2id[query_idx]
            
            top_ann_pid = I_nearest_neighbor[query_idx, :].copy()
            scores = I_scores[query_idx, :].copy()
            selected_ann_idx = top_ann_pid[:num_results] #only take top num_results from each shard. this can be lower than self.num_results for unsafe retrieval
            rank = 0
            seen_pid = set()
            
            for i, idx in enumerate(selected_ann_idx):
                pred_pid = passage_embedding2id[idx]

                if pred_pid not in seen_pid:
                    # this check handles multiple vector per document
                    rank += 1
                    docno = self.docid2docno[pred_pid+offset]
                    rtr.append([query_id, qid2q[query_id], pred_pid, docno, rank, scores[i]])
                    seen_pid.add(pred_pid)
        return pd.DataFrame(rtr, columns=["qid", "query", "docid", "docno", "rank", "score"])


class ANCETextScorer(TransformerBase):

    def __init__(self, checkpoint_path=None, text_field='text', **kwargs):
        self.args = type('', (), {})()
        args = self.args
        args.local_rank = -1
        args.model_type = 'rdot_nll'
        args.cache_dir  = None
        args.no_cuda = False
        args.max_query_length = 64
        args.max_seq_length = 128
        args.per_gpu_eval_batch_size = 128
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.__dict__.update(kwargs)

        config, tokenizer, model = load_model(self.args, checkpoint_path)
        self.model = model
        self.tokenizer = tokenizer
        self.text_field = text_field

    def __str__(self):
        return "ANCETextScorer"

    def transform(self, df):
        queries=[]
        docs = []
        idx_by_query = {}
        query_idxs = []
        # We do not want to redo the calculation of query representations, but due to logging
        # in the ance package, doing a groupby or pt.apply.by_query here will result in
        # excessive log messages. So we instead calculate each query rep once and keep track of
        # the correspeonding index so we can project back out the original sequence
        for q in df["query"].to_list():
            if q in idx_by_query:
                query_idxs.append(idx_by_query[q])
            else:
                passage = self.tokenizer.encode(
                    q,
                    add_special_tokens=True,
                    max_length=self.args.max_seq_length,
                )
                    
                passage_len = min(len(passage), self.args.max_query_length)
                input_id_b = pad_input_ids(passage, self.args.max_query_length)
                queries.append([passage_len, input_id_b])
                qidx = len(idx_by_query)
                idx_by_query[q] = qidx
                query_idxs.append(qidx)

        for d in df[self.text_field].to_list():
            passage = self.tokenizer.encode(
                d,
                add_special_tokens=True,
                max_length=self.args.max_seq_length,
            )
                
            passage_len = min(len(passage), self.args.max_seq_length)
            input_id_b = pad_input_ids(passage, self.args.max_seq_length)
            docs.append([passage_len, input_id_b])
        
        query_embeddings, _ = StreamInferenceDoc(self.args, self.model, GetProcessingFn(
             self.args, query=True), "transform", queries, is_query_inference=True)

        passage_embeddings, _ = StreamInferenceDoc(self.args, self.model, GetProcessingFn(
            self.args, query=False), "transform", docs, is_query_inference=False)

        # project out the query representations (see comment above)
        query_embeddings = query_embeddings[query_idxs]

        scores = (query_embeddings * passage_embeddings).sum(axis=1)

        return df.assign(score=scores)
