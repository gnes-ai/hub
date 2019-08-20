FROM pytorch/pytorch

LABEL maintainer="team@gnes.ai"

RUN pip install -U pytorch-transformers gnes --no-cache-dir --compile

ADD *.py *.yml ./

# [Optional] run a simple unit test
# you probably want to comment out this line in the CICD pipeline,
# as your CI runner for "docker build" may not have enough memory
RUN python -m unittest test_transformer.py -v

ENTRYPOINT ["gnes", "encode", "--yaml_path", "transformer.yml", "--py_path", "transformer.py", "--read_only"]