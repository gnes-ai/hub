# 🚢 GNES Hub - Preprocessor

To create a new preprocessor, you can create a new class inherited from 
 
- `gnes.component.BasePreprocessor`
- `gnes.component.BaseImagePreprocessor`
- `gnes.component.BaseTextPreprocessor`
- `gnes.component.BaseAudioPreprocessor`
- `gnes.component.BaseVideoPreprocessor`
- `gnes.component.PipelinePreprocessor`
- `gnes.component.UnaryPreprocessor`


and override the following methods:

- `apply`

In general, you can always override the following methods:

- `__init__`
- `post_init`
- `__getstate__`
- `__setstate__`
- `train`