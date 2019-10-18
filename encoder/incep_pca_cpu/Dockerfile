FROM tensorflow/tensorflow:1.14.0-py3

LABEL maintainer="team@gnes.ai"

RUN apt-get update && apt-get install -y wget git \
	&& rm -rf /var/lib/apt/lists/*

#RUN pip install -U gnes --no-cache-dir --compile
RUN pip install git+https://github.com/gnes-ai/gnes.git@master sklearn

ADD inception.py test_inception.py inception.yml ./

ADD inception_cores ./inception_cores

#RUN apt-get update && apt-get install -y wget \
#	&& rm -rf /var/lib/apt/lists/*

RUN cd /; mkdir inception_v4 \
	&& cd /inception_v4 \
	&& wget -q http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz \
	&& tar zxf inception_v4_2016_09_09.tar.gz; rm inception_v4_2016_09_09.tar.gz

RUN python -m unittest test_inception.py -v

ENTRYPOINT ["gnes", "encode", "--yaml_path", "inception.yml", "--py_path", "inception_cores/inception_utils.py", "inception_cores/inception_v4.py", "inception.py", "--read_only"]