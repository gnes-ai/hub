#  Tencent is pleased to support the open source community by making GNES available.
#
#  Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from gnes.component import BaseVideoPreprocessor
from gnes.preprocessor.io_utils import gif
from gnes.proto import gnes_pb2, blob2array


class GifConverterPreprocessor(BaseVideoPreprocessor):
    def __init__(self, fps: int = 10, pix_fmt: str = 'rgb24', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pix_fmt = pix_fmt
        self.fps = fps

    def apply(self, doc: 'gnes_pb2.Document') -> None:
        super().apply(doc)
        if len(doc.chunks) > 0:
            for chunk in doc.chunks:
                images = blob2array(chunk.blob)
                chunk.raw = gif.encode_gif(
                    images, pix_fmt=self.pix_fmt, fps=self.fps)
        elif doc.raw_video:
            images = blob2array(doc.raw_video)
            doc.raw_bytes = gif.encode_gif(
                images, pix_fmt=self.pix_fmt, fps=self.fps)
        else:
            self.logger.error(
                'bad document: "doc.chunks" and "doc.raw_video" is empty!')
