import copy
import os
import unittest

import numpy as np
from gnes.proto import gnes_pb2, array2blob

from gif_converter import GifConverterPreprocessor


class TestGifConverter(unittest.TestCase):
    def setUp(self):
        self.dirname = os.path.dirname(__file__)
        self.yaml_path = os.path.join(self.dirname, 'gif_converter.yml')
        self.dump_path = os.path.join(self.dirname, 'converter.bin')
        self.frames_path = os.path.join(self.dirname, 'frames.npy')
        self.gif_converter = GifConverterPreprocessor.load_yaml(self.yaml_path)

        self.video_frames = np.load(self.frames_path)


    def test_gif_converter(self):

        raw_data = array2blob(self.video_frames)

        doc = gnes_pb2.Document()
        doc.doc_type = gnes_pb2.Document.VIDEO
        doc.raw_video.CopyFrom(raw_data)
        self.gif_converter.apply(doc)
        doc1 = copy.deepcopy(doc)

        doc = gnes_pb2.Document()
        doc.doc_type = gnes_pb2.Document.VIDEO
        chunk = doc.chunks.add()
        chunk.blob.CopyFrom(raw_data)
        self.gif_converter.apply(doc)
        doc2 = copy.deepcopy(doc)

        self.assertEqual(doc1.raw_bytes, doc2.chunks[0].raw)

    def test_dump_load(self):
        raw_data = array2blob(self.video_frames)

        doc = gnes_pb2.Document()
        doc.doc_type = gnes_pb2.Document.VIDEO
        doc.raw_video.CopyFrom(raw_data)
        self.gif_converter.apply(doc)
        doc1 = copy.deepcopy(doc)

        self.gif_converter.dump(self.dump_path)
        converter = GifConverterPreprocessor.load(self.dump_path)

        doc = gnes_pb2.Document()
        doc.doc_type = gnes_pb2.Document.VIDEO
        chunk = doc.chunks.add()
        chunk.blob.CopyFrom(raw_data)
        converter.apply(doc)
        doc2 = copy.deepcopy(doc)

        self.assertEqual(doc1.raw_bytes, doc2.chunks[0].raw)


    def tearDown(self):
        if os.path.exists(self.dump_path):
            os.remove(self.dump_path)
