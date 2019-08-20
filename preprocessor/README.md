# 🚢 GNES Hub - Preprocessor

To create a new preprocessor, you can create a new class inherited from `gnes.preprocessor.base.BasePreprocessor` and override the following methods:

- `apply`

In general, you can always override the following methods:

- `__init__`
- `post_init`
- `__getstate__`
- `__setstate__`
- `train`