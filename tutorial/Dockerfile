FROM pytorch/pytorch

LABEL maintainer="team@gnes.ai"

RUN pip install -U pytorch-transformers gnes --no-cache-dir --compile

ENV WORKSPACE="/workspace"
WORKDIR $WORKSPACE

# register the python module
ENV GNES_CONTRIB_MODULE="$WORKSPACE/transformer.py"

ADD *.py *.yml ./

# [Optional] run a simple unit test
# you probably want to comment out this line in the CICD pipeline,
# as your CI runner for "docker build" may not have enough memory
RUN python -m unittest test_transformer.py -v

ENTRYPOINT ["gnes", "encode", "--yaml_path", "transformer.yml", "--read_only"]