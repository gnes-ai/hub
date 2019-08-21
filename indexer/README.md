# 🚢 GNES Hub - Indexer

To create a new indexer, you can create a new class inherited from 

- `gnes.component.BaseVectorIndexer`
- `gnes.component.BaseIndexer`
- `gnes.component.BaseTextIndexer`
- `gnes.component.BaseKeyIndexer`
- `gnes.component.JointIndexer`
 
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