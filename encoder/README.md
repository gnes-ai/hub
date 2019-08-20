# 🚢 GNES Hub - Encoder

To create a new indexer, you can create a new class inherited from 

- `gnes.encoder.base.BaseEncoder`
- `gnes.encoder.base.BaseImageEncoder`
- `gnes.encoder.base.BaseVideoEncoder`
- `gnes.encoder.base.BaseTextEncoder`
- `gnes.encoder.base.BaseNumericEncoder`
- `gnes.encoder.base.BaseAudioEncoder`
- `gnes.encoder.base.BaseBinaryEncoder`

and then override the following methods:

- `encode`

In general, you can always override the following methods:

- `__init__`
- `post_init`
- `__getstate__`
- `__setstate__`
- `train`