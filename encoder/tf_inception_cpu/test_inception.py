import numpy as np
import os
import unittest

from gnes.encoder.base import BaseEncoder

class TestTFInceptionEncoder(unittest.TestCase):

    def setUp(self):
        dirname = os.path.dirname(__file__)
        self.dump_path = os.path.join(dirname, 'model.bin')
        # one image with two chunks
        self.test_img = [[np.random.randint(0, 255, (299, 299, 3)).astype('uint8'),
                            np.random.randint(0, 255, (299, 299, 3)).astype('uint8')]]
        self.yaml_path = os.path.join(dirname, 'inception.yml')

    def test_inception_encoding(self):
        self.encoder = BaseEncoder.load_yaml(self.yaml_path)
        for test_img in self.test_img:
            vec = self.encoder.encode(test_img)
            self.assertEqual(vec.shape[0], 2)
            self.assertEqual(vec.shape[1], 1536)

    def test_dump_load(self):
        self.encoder = BaseEncoder.load_yaml(self.yaml_path)

        self.encoder.dump(self.dump_path)

        encoder2 = BaseEncoder.load(self.dump_path)

        for test_img in self.test_img:
            vec = encoder2.encode(test_img)
            self.assertEqual(vec.shape[0], 2)
            self.assertEqual(vec.shape[1], 1536)

    def tearDown(self):
        if os.path.exists(self.dump_path):
            os.remove(self.dump_path)