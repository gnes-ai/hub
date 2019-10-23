from typing import List
from collections import defaultdict, OrderedDict

from gnes.router.base import BaseReduceRouter
from gnes.proto import gnes_pb2
from gnes.helper import batching

from transformers import *
import torch
import numpy as np

class RerankRouter(BaseReduceRouter):

    def __init__(self, model_name: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def post_init(self):
        model_config = AutoConfig.from_pretrained(self.model_name)
        model_config.num_labels = 1 # set up for regression
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                               config=model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def get_key(self, x: 'gnes_pb2.Response.QueryResponse.ScoredResult') -> str:
        return x.doc.doc_id

    def set_key(self, x: 'gnes_pb2.Response.QueryResponse.ScoredResult', k: str) -> None:
        x.doc.doc_id = k

    @batching
    def apply(self, msg: 'gnes_pb2.Message', accum_msgs: List['gnes_pb2.Message'], *args, **kwargs):
        # now convert chunk results to doc results
        all_scored_results = [sr for m in accum_msgs for sr in m.response.search.topk_results]
        score_dict = defaultdict(list)

        ids = [self.tokenizer.encode(sr.doc.raw_text) for sr in all_scored_results]
        max_len = max(len(t) for t in ids)
        ids = [t + [0] * (max_len - len(t)) for t in ids]
        input_ids = torch.tensor(ids)
        with torch.no_grad():
            logits = self.rerank_model(input_ids)[0]  # Models outputs are now tuples
            scores = np.squeeze(logits.detach().cpu().numpy())

        # count score by iterating over chunks
        for c, score in zip(all_scored_results, scores):
            score_dict[self.get_key(c)].append(score)

        for k, v in score_dict.items():
            score_dict[k] = sum(v)

        k = msg.response.search.top_k
        score_dict = OrderedDict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:k])

        msg.response.search.ClearField('topk_results')

        for k, v in score_dict.items():
            r = msg.response.search.topk_results.add()
            r.score.value = float(v)
            self.set_key(r, k)

        super().apply(msg, accum_msgs)