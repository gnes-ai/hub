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
                self.logger.info("the length of shape of images: %d" % len(images.shape))
                self.logger.info("the first dimension of images: %d" % images.shape[0])
                self.logger.info("the second dimension of images: %d" % images.shape[1])
                self.logger.info("the third dimension of images: %d" % images.shape[2])
                self.logger.info("the fourth dimension of images: %d" % images.shape[3])
                
                images = np.array([np.array(Image.fromarray(img.astype('uint8')).resize((self.target_width, self.target_height))) 
                                            for img in images])
                if len(images) == 0:
                    self.logger.warning("this chunk has no frame!")
                elif self.sframes == 1:
                    idx = [int(len(images) / 2)]
                    chunk.blob.CopyFrom(array2blob(images[idx]))
                elif self.sframes > 0 and len(images) > self.sframes:
                    if len(images) >= 2 * self.sframes:
                        step = math.ceil(len(images) / self.sframes)
                        chunk.blob.CopyFrom(array2blob(images[::step]))
                    else:
                        idx = np.sort(np.random.choice(len(images), self.sframes, replace=False))
                        chunk.blob.CopyFrom(array2blob(images[idx]))
                del images
        else:
            self.logger.error(
                'bad document: "doc.chunks" is empty!')
