FROM pytorch/pytorch

RUN pip install -U pytorch-transformers gnes --no-cache-dir --compile

ADD *.py *.yml ./

ENTRYPOINT ["gnes", "route", "--yaml_path", "rerank_transformers.yml", "--py_path", "rerank_transformers.py"]