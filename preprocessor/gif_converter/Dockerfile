FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /tmp/* \
    && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

ADD *.py *.yml *.npy ./

RUN python -m unittest test_converter.py -v

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "gif_converter.yml", "--py_path", "gif_converter.py"]