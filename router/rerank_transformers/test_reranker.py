import os
import unittest

from gnes.proto import gnes_pb2
from gnes.client.base import ZmqClient
from gnes.service.base import SocketType
from gnes.cli.parser import set_router_parser, _set_client_parser
from gnes.service.router import RouterService


class TestReranker(unittest.TestCase):

    def setUp(self):
        dirname = os.path.dirname(__file__)
        self.rerank_router_yaml = os.path.join(dirname,'rerank_transformers.yml')
        self.python_code = os.path.join(dirname, 'rerank_transformers.py')

        self.test_str = []
        with open(os.path.join(dirname, 'sonnets_small.txt')) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.test_str.append(line)

    def test_rerank(self):
        args = set_router_parser().parse_args([
            '--yaml_path', self.rerank_router_yaml,
            '--socket_out', str(SocketType.PUB_BIND),
            '--py_path', self.python_code
        ])
        c_args = _set_client_parser().parse_args([
            '--port_in', str(args.port_out),
            '--port_out', str(args.port_in),
            '--socket_in', str(SocketType.SUB_CONNECT)
        ])
        with RouterService(args), ZmqClient(c_args) as c1, ZmqClient(c_args) as c2:
            msg = gnes_pb2.Message()
            msg.response.search.ClearField('topk_results')

            for i, line in enumerate(self.test_str):
                s = msg.response.search.topk_results.add()
                s.score.value = 0.1
                s.doc.doc_id = i
                s.doc.raw_text = line

            msg.envelope.num_part.extend([1])
            msg.response.search.top_k = 5
            c1.send_message(msg)

            r = c1.recv_message()
            self.assertSequenceEqual(r.envelope.num_part, [1])
            self.assertEqual(len(r.response.search.topk_results), 5)

            msg = gnes_pb2.Message()
            msg.response.search.ClearField('topk_results')

            for i, line in enumerate(self.test_str[:3]):
                s = msg.response.search.topk_results.add()
                s.score.value = 0.1
                s.doc.doc_id = i
                s.doc.raw_text = line

            msg.envelope.num_part.extend([1])
            msg.response.search.top_k = 5
            c1.send_message(msg)

            r = c1.recv_message()
            self.assertSequenceEqual(r.envelope.num_part, [1])
            self.assertEqual(len(r.response.search.topk_results), 3)

    def tearDown(self):
        pass