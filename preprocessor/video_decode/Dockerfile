FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN apt-get update -y && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /tmp/* \
    && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /video_decode

ADD *.py *.yml ./

ADD video ./video

RUN python -m unittest test_video_decode.py -v

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "video_decode.yml", "--py_path", "video_decode.py", "--read_only"]
