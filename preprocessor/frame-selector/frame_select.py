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

import numpy as np
import math
from PIL import Image

from gnes.preprocessor.base import BaseVideoPreprocessor
from gnes.proto import gnes_pb2, array2blob, blob2array


class FrameSelectPreprocessor(BaseVideoPreprocessor):

    def __init__(self,
                 sframes: int = 1,
                 target_width: int = 299,
                 target_height: int = 299,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sframes = sframes
        self.target_width = target_width
        self.target_height = target_height

    def apply(self, doc: 'gnes_pb2.Document') -> None:
        super().apply(doc)
        if len(doc.chunks) > 0:
            for chunk in doc.chunks:
                images = blob2array(chunk.blob)

                if len(images) == 0:
                    self.logger.warning("this chunk has no frame!")
                elif self.sframes == 1:
                    idx = int(len(images) / 2)
                    frame = np.array(Image.fromarray(images[idx].astype('uint8')).resize((self.target_width, self.target_height)))
                    frame = np.expand_dims(frame, axis=0)
                    # self.logger.info("choose one frame, the shape is: (%d, %d, %d, %d)" % (
                    #     frame.shape[0], frame.shape[1], frame.shape[2], frame.shape[3]
                    # ))
                    chunk.blob.CopyFrom(array2blob(frame))
                elif self.sframes > 0 and len(images) > self.sframes:
                    if len(images) >= 2 * self.sframes:
                        step = math.ceil(len(images) / self.sframes)
                        frames = images[::step]
                    else:
                        idx = np.sort(np.random.choice(len(images), self.sframes, replace=False))
                        frames = images[idx]

                    frames = np.array(
                        [np.array(Image.fromarray(img.astype('uint8')).resize((self.target_width, self.target_height)))
                         for img in frames])
                    chunk.blob.CopyFrom(array2blob(frames))
                del images
        else:
            self.logger.error(
                'bad document: "doc.chunks" is empty!')
