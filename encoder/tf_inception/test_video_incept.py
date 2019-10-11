import numpy as np
import os
import unittest

from gnes.encoder.base import BaseEncoder


class TestInceptEncoder(unittest.TestCase):
    def setUp(self):
        dirname = os.path.dirname(__file__)
        self.dump_path = os.path.join(dirname, 'model.bin')
        self.test_data = []
        self.test_chunk_size = [3, 4, 5]

        for n in self.test_chunk_size:
            x = np.random.randint(0, 255, (n, 299, 299, 3)).astype('uint8')
            self.test_data.append(x)
        self.yaml_path = os.path.join(dirname, 'video_inception.yml')

        self.encoder = BaseEncoder.load_yaml(self.yaml_path)

    def test_encode(self):
        encodes = self.encoder.encode(self.test_data)
        self.assertEqual(len(encodes), len(self.test_data))
        self.assertEqual([x.shape[0] for x in encodes], self.test_chunk_size)


    def test_dump_load(self):
        self.encoder.dump(self.dump_path)

        encoder2 = BaseEncoder.load(self.dump_path)

        encodes = encoder2.encode(self.test_data)
        self.assertEqual(len(encodes), len(self.test_data))
        self.assertEqual([x.shape[0] for x in encodes], self.test_chunk_size)

    def tearDown(self):
        if os.path.exists(self.dump_path):
            os.remove(self.dump_path)
