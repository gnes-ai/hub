import os
import unittest
import numpy as np

from gnes.encoder.base import BaseVideoEncoder


class TestI3dEncoder(unittest.TestCase):
    def setUp(self):
        dirname = os.path.dirname(__file__)
        self.test_video = [np.random.rand(2, 224, 224, 3).astype(np.uint8),
                         np.random.rand(4, 224, 224, 3).astype(np.uint8),
                         np.random.rand(6, 224, 224, 3).astype(np.uint8)]
        self.i3d_yaml = os.path.join(dirname, 'i3d_encoder.yml')

    def test_empty_service(self):
        args = set_encoder_parser().parse_args([
            '--yaml_path', self.i3d_yaml
        ])
        with ServiceManager(EncoderService, args):
            pass

    # @unittest.skip
    # def test_i3d_encoding(self):
    #     self.encoder = BaseVideoEncoder.load_yaml(self.i3d_yaml)
    #     vec = self.encoder.encode(self.test_video)

    #     self.assertEqual(vec.shape[0], len(self.test_video))
    #     print(vec.shape)
