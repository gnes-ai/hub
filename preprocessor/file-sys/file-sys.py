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


import os
from typing import List
import numpy as np

from gnes.preprocessor.base import BaseVideoPreprocessor
from gnes.proto import gnes_pb2, blob2array


class DirectoryPreprocessor(BaseVideoPreprocessor):

    def __init__(self, data_path: str,
                 keep_na_doc: bool = True,
                 file_suffix: str = 'mp4',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.file_suffix = file_suffix
        self.keep_na_doc = keep_na_doc
        self._NOT_FOUND = None

    def apply(self, docs: 'gnes_pb2.Document') -> None:
        """
        write GIFs of each document into disk
        folder structure: /data_path/doc_id/0.gif, 1.gif...
        :param docs: docs
        """
        dirs = os.path.join(self.data_path, str(docs.doc_id))
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        # keep doc meta in .meta file
        with open(os.path.join(dirs, '.meta'), 'wb') as f:
            f.write(docs.meta_info or b'')
            self.logger.info("successfully write meta info for: %s" % str(docs.doc_id))

        self.logger.info("%s has total %d chunks." % (str(docs.doc_id), len(docs.chunks)))
        for i, chunk in enumerate(docs.chunks):
            data_type = chunk.WhichOneof('content')
            if data_type == 'raw':
                with open(os.path.join(dirs, '%d.%s' % (i, self.file_suffix)), 'wb') as f:
                    f.write(chunk.raw)
            elif data_type == 'blob':
                np.save(os.path.join(dirs, '%d' % i), blob2array(chunk.blob))
                self.logger.info("successfully write blob %d for: %s" % (i, str(docs.doc_id)))
            else:
                self.logger.info("data_type is : %s" % str(data_type))
                raise NotImplementedError

