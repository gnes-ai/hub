#  Tencent is pleased to support the open source community by making GNES available.
#
#  Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from typing import List

import numpy as np
import torch
from gnes.encoder.base import BaseTextEncoder
from gnes.helper import batching
from pytorch_transformers import *


class PyTorchTransformers(BaseTextEncoder):

    def __init__(self, model_name: str = 'bert-base-uncased', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def post_init(self):
        # select the model, tokenizer & weight accordingly
        model_class, tokenizer_class, pretrained_weights = \
            {k[-1]: k for k in
             [(BertModel, BertTokenizer, 'bert-base-uncased'),
              (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
              (GPT2Model, GPT2Tokenizer, 'gpt2'),
              (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
              (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
              (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024'),
              (RobertaModel, RobertaTokenizer, 'roberta-base')]}[self.model_name]

        def load_model_tokenizer(x):
            return model_class.from_pretrained(x), tokenizer_class.from_pretrained(x)

        try:
            self.model, self.tokenizer = load_model_tokenizer(self.work_dir)
        except Exception:
            self.logger.warning('cannot deserialize model/tokenizer from %s, will download from web' % self.work_dir)
            self.model, self.tokenizer = load_model_tokenizer(pretrained_weights)

    @batching
    def encode(self, text: List[str], *args, **kwargs) -> np.ndarray:
        # encoding and padding
        ids = [self.tokenizer.encode(t) for t in text]
        max_len = max(len(t) for t in ids)
        ids = [t + [0] * (max_len - len(t)) for t in ids]
        input_ids = torch.tensor(ids)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        self.logger.info(last_hidden_states)
        return np.array(last_hidden_states)

    def __getstate__(self):
        self.model.save_pretrained(self.work_dir)
        self.tokenizer.save_pretrained(self.work_dir)
        return super().__getstate__()
