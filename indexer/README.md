# 🚢 GNES Hub - Indexer

To create a new indexer, you can create a new class inherited from 

- `gnes.indexer.base.BaseIndexer`
- `gnes.indexer.base.BaseVectorIndexer`
- `gnes.indexer.base.BaseTextIndexer`
- `gnes.indexer.base.BaseKeyIndexer`
 
and then override the following methods:

- `add`
- `query`
- `normalize_score`

In general, you can always override the following methods:

- `__init__`
- `post_init`
- `__getstate__`
- `__setstate__`
- `train`