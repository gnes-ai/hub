# 🚢 GNES Hub - Router

To create a new router, you can create a new class inherited from 

- `gnes.component.BaseRouter`
- `gnes.component.BaseMapRouter`
- `gnes.component.BaseReduceRouter`

and then override the following methods:

- `apply`

In general, you can always override the following methods:

- `__init__`
- `post_init`
- `__getstate__`
- `__setstate__`
- `train`