import copy
import os
import unittest

import numpy as np
from gnes.proto import gnes_pb2, array2blob

from gnes.indexer.base import BaseDocIndexer
from gnes.helper import PathImporter


class TestVideoShotIndexer(unittest.TestCase):
    def setUp(self):
        self.dirname = os.path.dirname(__file__)
        self.yaml_path = os.path.join(self.dirname, 'videoshot.indexer.yml')
        self.dump_path = os.path.join(self.dirname, 'videoshot_indexer.bin')
        self.frames_path = os.path.join(self.dirname, 'test_frames.npy')

        self.video_frames = np.load(self.frames_path)
        PathImporter.add_modules('video_shot_indexer.py')
        self.indexer = BaseDocIndexer.load_yaml(self.yaml_path)

    def test_videoshot_indexer(self):
        raw_data = array2blob(self.video_frames)

        doc = gnes_pb2.Document()
        doc.doc_type = gnes_pb2.Document.VIDEO
        chunk = doc.chunks.add()
        chunk.blob.CopyFrom(raw_data)

        self.indexer.add([0], [doc])

    def test_dump_load(self):
        raw_data = array2blob(self.video_frames)

        doc = gnes_pb2.Document()
        doc.doc_type = gnes_pb2.Document.VIDEO
        chunk = doc.chunks.add()
        chunk.blob.CopyFrom(raw_data)

        doc1 = copy.deepcopy(doc)

        self.indexer.dump(self.dump_path)

        indexer = BaseDocIndexer.load(self.dump_path)

        indexer.add([0], [doc1])

    def tearDown(self):
        if os.path.exists(self.dump_path):
            os.remove(self.dump_path)
