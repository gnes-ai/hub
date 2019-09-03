FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN apt-get update -y && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /tmp/* && rm -rf /gnes \
    && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /gif_decode

ADD *.py *.yml ./

ADD gifs ./gifs

RUN python -m unittest test_gif_decode.py -v

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "gif_decode.yml", "--py_path", "gif_decode.py", "--read_only"]
