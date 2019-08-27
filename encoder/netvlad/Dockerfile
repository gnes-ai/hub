FROM tensorflow/tensorflow:1.14.0-gpu-py3

LABEL maintainer="team@gnes.ai"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8



ADD ./inception_v4.ckpt /
ADD ./netvlad /netvlad

RUN apt-get update && apt-get install wget
RUN pip install Pillow 

RUN wget https://github.com/gnes-ai/gnes/archive/master.zip && \
    unzip master.zip && \
    cd gnes-master && pip install . -U 


ADD encoder.yml /

ENTRYPOINT ["gnes", "encode", "--yaml_path", "/encoder.yml"]