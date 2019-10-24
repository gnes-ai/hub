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

from typing import List

import numpy as np

from gnes.encoder.base import BaseVideoEncoder
from gnes.helper import batching, get_first_available_gpu


class I3dEncoder(BaseVideoEncoder):
    batch_size = 1

    def __init__(self, model_dir: str,
                 output_layer: str,
                 num_classes: int = 400,
                 frame_size_x: int = 224,
                 frame_size_y: int = 224,
                 num_frame_per_clib: int = 16,
                 rgb_channels: int = 3,
                 on_gpu: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dir = model_dir
        self.output_layer = output_layer
        self.num_classes = num_classes
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.num_frame_per_clib = num_frame_per_clib
        self.rgb_channels = rgb_channels
        self.on_gpu = on_gpu

    def post_init(self):
        import tensorflow as tf
        from i3d_cores.i3d import InceptionI3d

        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(get_first_available_gpu())

        with tf.Graph().as_default():
            self.rgb_images_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,
                                                                             self.num_frame_per_clib,
                                                                             self.frame_size_x,
                                                                             self.frame_size_y,
                                                                             self.rgb_channels))
            is_training = False

            with tf.variable_scope('RGB'):
                self.feature, _ = InceptionI3d(
                    num_classes=self.num_classes,
                    spatial_squeeze=True,
                    final_endpoint=self.output_layer,
                    name='inception_i3d'
                )(self.rgb_images_placeholder, is_training)
            init = tf.global_variables_initializer()

            config = tf.ConfigProto(log_device_placement=False)
            if self.on_gpu:
                config.gpu_options.allow_growth = True

            self.sess = tf.Session(config=config)
            self.sess.run(init)

            checkpoint_file = self.model_dir
            meta_graph_location = self.model_dir + '.meta'
            saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
            saver.restore(self.sess, checkpoint_file)

    def encode(self, data: List['np.ndarray'], *args, **kwargs) -> np.ndarray:
        def _padding(data):
            _data = np.array(
                [np.concatenate((d, np.zeros((self.num_frame_per_clib - d.shape[0],
                                              self.frame_size_x,
                                              self.frame_size_y,
                                              self.rgb_channels), dtype=np.float32)), axis=0)
                 if d.shape[0] < self.num_frame_per_clib else d[:self.num_frame_per_clib] for d in data])
            return _data

        @batching
        def _encode(_, data):
            feature, = self.sess.run([self.feature], feed_dict={self.rgb_images_placeholder: data})
            return np.array(feature).astype(np.float32)

        return _encode(self, _padding(data))
