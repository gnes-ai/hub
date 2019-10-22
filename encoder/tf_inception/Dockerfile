FROM tensorflow/tensorflow:1.14.0-gpu-py3

LABEL maintainer="team@gnes.ai"

RUN apt-get update && apt-get install -y wget git && \
    apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN cd /; mkdir inception_v4 \
    && cd /inception_v4 \
    && wget -q http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz \
    && tar zxf inception_v4_2016_09_09.tar.gz; rm inception_v4_2016_09_09.tar.gz

RUN pip install pillow GPUtil  --no-cache-dir

#RUN pip install -U gnes --no-cache-dir --compile
RUN echo "1"
RUN pip install git+https://github.com/gnes-ai/gnes.git@master --compile


ADD test_video_incept.py video_inception.yml ./
RUN python -m unittest test_video_incept.py -v

ENTRYPOINT ["gnes"]