import numpy as np
import os
import unittest

#from gnes.encoder.base import BaseEncoder
from gnes.cli.parser import set_encoder_parser, _set_client_parser
from gnes.client.base import ZmqClient
from gnes.proto import gnes_pb2, RequestGenerator, blob2array, array2blob
from gnes.service.base import ServiceManager
from gnes.service.encoder import EncoderService


class TestPCALocalEncoder(unittest.TestCase):

    def setUp(self):
        dirname = os.path.dirname(__file__)
        self.dump_path = os.path.join(dirname, 'model.bin')
        # 1000 samples with data_dim 1024
        self.test_numeric = np.random.randint(0, 255, (1000, 1024)).astype('float32')
        self.pcalocal_yaml_path = 'pcalocal.yml'
        self.pca_yaml_path = os.path.join(dirname, 'pca.yml')

    def test_pca_local_encoding(self):
        args = set_encoder_parser().parse_args(
            ['--yaml_path', self.pcalocal_yaml_path, '--py_path', 'pca.py'])
        c_args = _set_client_parser().parse_args([
            '--port_in',
            str(args.port_out), '--port_out',
            str(args.port_in)])

        with ServiceManager(EncoderService, args), ZmqClient(c_args) as client:
            msg = gnes_pb2.Message()
            d = msg.request.train.docs.add()
            d.doc_id = 0
            d.raw_bytes = self.test_numeric.tobytes()
            d.doc_type = gnes_pb2.Document.TEXT
            msg.request.train.flush = True
            msg.envelope.num_part.append(1)

            c = d.chunks.add()
            c.doc_id = d.doc_id
            c.blob.CopyFrom(array2blob(self.test_numeric))
            c.offset = 0
            c.weight = 1.0

            client.send_message(msg)
            r = client.recv_message()

        '''with ServiceManager(EncoderService, args), ZmqClient(c_args) as client:
            for req in RequestGenerator.index(self.test_numeric):
                msg = gnes_pb2.Message()
                msg.request.index.CopyFrom(req.index)
                client.send_message(msg)
                r = client.recv_message()
                for d in r.request.index.docs:
                    self.assertGreater(len(d.chunks), 0)
                    for _ in range(len(d.chunks)):
                        shape = blob2array(d.chunks[_].blob).shape
                        self.assertEqual(shape, (1000, 300))'''

    '''def test_pca_local_encoding(self):
        self.encoder = BaseEncoder.load_yaml(self.pcalocal_yaml_path)
        # train before encode to create pca_components
        self.encoder.train(self.test_numeric)
        vec = self.encoder.encode(self.test_numeric)
        self.assertEqual(vec.shape, (1000, 300))
        # dump after train with valied pca_components
        self.encoder.dump(self.dump_path)
        encoder2 = BaseEncoder.load(self.dump_path)
        vec = encoder2.encode(self.test_numeric)
        self.assertEqual(vec.shape, (1000, 300))

    def test_pca_encoding(self):
        self.encoder = BaseEncoder.load_yaml(self.pca_yaml_path)
        # train before encode to create pca_components
        self.encoder.train(self.test_numeric)
        vec = self.encoder.encode(self.test_numeric)
        self.assertEqual(vec.shape, (1000, 300))
        # dump after train with valied pca_components
        self.encoder.dump(self.dump_path)
        encoder2 = BaseEncoder.load(self.dump_path)
        vec = encoder2.encode(self.test_numeric)
        self.assertEqual(vec.shape, (1000, 300))

    def tearDown(self):
        if os.path.exists(self.dump_path):
            os.remove(self.dump_path)'''