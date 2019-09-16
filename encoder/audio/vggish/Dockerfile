FROM tensorflow/tensorflow:latest-gpu-py3

LABEL maintainer="team@gnes.ai"

RUN pip install -U gnes --no-cache-dir --compile

ADD *.py *.yml ./

VOLUME /model

ENTRYPOINT ["gnes", "encode", "--yaml_path", "encoder.vggish.yml", "--read_only"]