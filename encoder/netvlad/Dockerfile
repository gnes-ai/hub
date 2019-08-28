FROM tensorflow/tensorflow:1.14.0-gpu-py3

LABEL maintainer="team@gnes.ai"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ADD ./netvlad /netvlad

RUN apt-get update && apt-get install wget git -y

RUN wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz && \
    tar -xvf inception_v4_2016_09_09.tar.gz 


RUN pip install Pillow 
RUN pip install https://github.com/gnes-ai/gnes/archive/master.zip

ADD encoder.yml /

ENTRYPOINT ["gnes", "encode", "--yaml_path", "/encoder.yml"]