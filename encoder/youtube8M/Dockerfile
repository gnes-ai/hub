FROM tensorflow/tensorflow:latest-gpu-py3

LABEL maintainer="team@gnes.ai"

RUN pip install -U gnes --no-cache-dir --compile

RUN pip install pillow

VOLUME /model

ADD *.py *.yml ./ 

ENTRYPOINT ["gnes", "encode", "--yaml_path", "encoder.yt8m.yml", "--read_only"]