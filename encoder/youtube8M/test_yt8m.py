import numpy as np
import os
import unittest

from gnes.encoder.base import BaseEncoder

class TestYT8MEncoder(unittest.TestCase):

    def setUp(self):
        dirname = os.path.dirname(__file__)
        self.dump_path = os.path.join(dirname, 'model.bin')
        # one image with two chunks
        self.test_frames = [np.random.rand(10, 299, 299, 3).astype(np.uint8),
                         np.random.rand(6, 299, 299, 3).astype(np.uint8),
                         np.random.rand(8, 299, 299, 3).astype(np.uint8)]
        self.yaml_path = os.path.join(dirname, 'encoder.yt8m.yml')

    def test_yt8m_encoding(self):
        self.encoder = BaseEncoder.load_yaml(self.yaml_path)
        vec = self.encoder.encode(self.test_frames)
        self.assertEqual(vec.shape[0], 3)
        self.assertEqual(vec.shape[1], 19310)

    def test_dump_load(self):
        self.encoder = BaseEncoder.load_yaml(self.yaml_path)

        self.encoder.dump(self.dump_path)

        encoder2 = BaseEncoder.load(self.dump_path)

        vec = encoder2.encode(self.test_frames)
        self.assertEqual(vec.shape[0], 3)
        self.assertEqual(vec.shape[1], 19310)

    def tearDown(self):
        if os.path.exists(self.dump_path):
            os.remove(self.dump_path)