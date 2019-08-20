# 🚢 GNES Hub - Router

To create a new router, you can create a new class inherited from 

- `gnes.router.base.BaseRouter`
- `gnes.router.base.BaseMapRouter`
- `gnes.router.base.BaseReduceRouter`

and then override the following methods:

- `apply`

In general, you can always override the following methods:

- `__init__`
- `post_init`
- `__getstate__`
- `__setstate__`
- `train`