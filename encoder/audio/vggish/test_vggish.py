import os
import unittest
import numpy as np

from gnes.encoder.audio.vggish import VggishEncoder


class TestVggishEncoder(unittest.TestCase):
    def setUp(self):
        self.dirname = os.path.dirname(__file__)
        self.audios = [np.random.rand(10, 96, 64),
                       np.random.rand(15, 96, 64),
                       np.random.rand(5, 96, 64)]
        self.vggish_yaml = os.path.join(self.dirname, 'encoder.vggish.yml')

    def test_vggish_encoding(self):
        self.encoder = VggishEncoder.load_yaml(self.vggish_yaml)
        vec = self.encoder.encode(self.audios)
        self.assertEqual(len(vec.shape), 2)
        self.assertEqual(vec.shape[0], len(self.audios))
        self.assertEqual(vec.shape[1], 128)