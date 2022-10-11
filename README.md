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

Note that, by default, this codebase uses exhaustive search when querying the dense index. This is not ideal for performance, but it is the setting contriever was evaluated on. If you want to switch to approximate search, you can do so by setting the `faiss_factory_config` attribute of `SentenceTransformersRetriever` / `SentenceTransformersIndexer` to any valid index factory string (or pass `faiss_factory_config=` to the `contriever_scifact.py` script). I recommend checking out [the faiss docs][3] for more info on the various approximate search options; a good starting point is probably `HNSW`:

```bash
python scripts/contriever_scifact.py \
    faiss_factory_config='HNSW32' \
    per_call_size=1024
```

This gets you close performance to the exact search:

```bash
                          name       map  recip_rank      P.10  ndcg_cut.10
0                         BM25  0.637799    0.647941  0.091667     0.683904
1  facebook/contriever-msmarco  0.629594    0.642171  0.090000     0.670841
```

Note Note that sometimes you might have to increment the number of passages batch batch (`per_call_size`); this is because the approximate search gets trained using the first batch of passages, and the more passages you have, the better the search will be.

In the example above, switching to `faiss_factory_config='HNSW64'` gets you another point of accuracy in nDCG@10, but it will increase query time.

[1]: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
[2]: https://arxiv.org/pdf/2112.09118.pdf
[3]: https://github.com/facebookresearch/faiss/wiki/The-index-factory
