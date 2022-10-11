# pyterrier_sentence_transformers
A codebase derived on `terrierteam/pyterrier_ance` that allows encoding using any `sentence_transformers` model.

## Installation

If running faiss on CPU:

```bash
pip install git+https://github.com/soldni/pyterrier_sentence_transformers.git
conda install -c pytorch faiss-gpu cudatoolkit=11.3
```
else, for gpu support:

```bash
pip install git+https://github.com/soldni/pyterrier_sentence_transformers.git
conda install -c pytorch faiss-cpu
```

If you need to install faiss from scratch, see [instructions here][1].


## Running

See example in `scripts/contriever_scifact.py`.

**Note**: I can't quite manage to reproduce the results from [Table 2][2] on the paper, but I'm not sure if it's due to the codebase or the model. I'm still investigating.

```bash
                  name       map  recip_rank      P.10  ndcg_cut.10
0                 BM25  0.626997    0.636829  0.090333     0.672167
1   contriever-msmarco  0.619778    0.631274  0.087667     0.655547
```

nDCG@10 should be `66.5` on row 0, and row 1 should be `67.7`.


[1]: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
[2]: https://arxiv.org/pdf/2112.09118.pdf
