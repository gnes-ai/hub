# 🚢 GNES Hub - Encoder

To create a new indexer, you can create a new class inherited from 

- `gnes.component.BaseEncoder`
- `gnes.component.BaseTextEncoder`
- `gnes.component.BaseAudioEncoder`
- `gnes.component.BaseImageEncoder`
- `gnes.component.BaseVideoEncoder`
- `gnes.component.BaseBinaryEncoder`
- `gnes.component.BaseNumericEncoder`
- `gnes.component.PipelineEncoder`

and then override the following methods:

- `encode`

In general, you can always override the following methods:

- `__init__`
- `post_init`
- `__getstate__`
- `__setstate__`
- `train`