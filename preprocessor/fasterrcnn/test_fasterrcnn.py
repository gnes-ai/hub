import os
import unittest
import zipfile

from gnes.proto import gnes_pb2, blob2array
from gnes.preprocessor.base import BasePreprocessor, PipelinePreprocessor


class TestProto(unittest.TestCase):

    def setUp(self):
        self.dirname = os.path.dirname(__file__)
        self.fasterrcnn_yaml = os.path.join(self.dirname, 'preprocessor.fasterrcnn.yml')
        self.model = BasePreprocessor.load_yaml(self.fasterrcnn_yaml)

    def test_segmentation_preprocessor(self):
        all_zips = zipfile.ZipFile('test.zip')
        all_bytes = [all_zips.open(v).read() for v in all_zips.namelist()]

        msg = gnes_pb2.Message()
        for pic_bytes in all_bytes:
            d = msg.request.index.docs.add()
            d.raw_bytes = pic_bytes
            self.model.apply(d)

            self.assertEqual(len(blob2array(d.chunks[0].blob).shape), 3)
            self.assertEqual(blob2array(d.chunks[0].blob).shape[-1], 3)
            self.assertEqual(blob2array(d.chunks[0].blob).shape[0], 224)
            self.assertEqual(blob2array(d.chunks[0].blob).shape[1], 224)
            print(blob2array(d.chunks[0].blob).dtype)