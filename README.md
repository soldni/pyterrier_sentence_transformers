# pyterrier_sentence_transformers
A codebase derived on `terrierteam/pyterrier_ance` that allows encoding using any `sentence_transformers` model.

## Installation

If running faiss on CPU:

```bash
pip install git+https://github.com/soldni/pyterrier_sentence_transformers.git
conda install -c pytorch faiss-cpu
```

else, for gpu support:

```bash
pip install git+https://github.com/soldni/pyterrier_sentence_transformers.git
conda install -c pytorch faiss-gpu cudatoolkit=11.3
```

If you need to install faiss from scratch, see [instructions here][1].


## Running

See example in `scripts/contriever_scifact.py`.

```bash
                          name       map  recip_rank      P.10  ndcg_cut.10
0                         BM25  0.637799    0.647941  0.091667     0.683904
1  facebook/contriever-msmarco  0.641346    0.653874  0.091667     0.682851
```

Note that the nDCG@10 we get for BM25 is much better than in the [paper][2]: instead of `66.5` on row 0, we get '68.4'. The contriever result is also a bit better, with `68.3` instead of `67.7`. Not sure what kind of magic pyterrier is doing here 🤷.


[1]: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
[2]: https://arxiv.org/pdf/2112.09118.pdf
