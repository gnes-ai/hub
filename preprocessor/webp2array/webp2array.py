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
import webp

from gnes.preprocessor.base import BaseVideoPreprocessor
from gnes.proto import gnes_pb2, array2blob


class Webp2ArrayPreprocessor(BaseVideoPreprocessor):
    def apply(self, doc: 'gnes_pb2.Document') -> None:
        super().apply(doc)

        # self.logger.info("doc id is: %s" % str(doc.doc_id))
        # self.logger.info("this doc has chunks: %d! for doc %s" % (len(doc.chunks), str(doc.doc_id)))

        total_frames = 0

        for offset, c in enumerate(doc.chunks):
            # self.logger.info("chunk offset is: %s" % str(offset))
            webp_data = webp.WebPData.from_buffer(c.raw)
            # self.logger.info("done transfer buffer! for chunk offset %s" % str(offset))
            dec = webp.WebPAnimDecoder.new(webp_data)
            # self.logger.info("done transfer webp! for chunk offset %s" % str(offset))

            image_list = []
            for arr, timestamp_ms in dec.frames():
                image = np.array(arr)[:, :, :-1].copy()
                image_list.append(image)

            # self.logger.info("done loading all frames! for chunk offset %s" % str(offset))
            c.offset = offset
            image_list_array = np.array(image_list)
            # self.logger.info("done transfer to numpy array! for chunk offset %s" % str(offset))
            c.blob.CopyFrom(array2blob(image_list_array))
            # self.logger.info("done transfer to blob! for chunk offset %s" % str(offset))
            total_frames += len(image_list)
            del dec
            # self.logger.info("done process webp! for chunk offset %s" % str(offset))

        # self.logger.info("this doc has frames: %d! for doc %s" % (total_frames, str(doc.doc_id)))

        for c in doc.chunks:
            c.weight /= total_frames



