FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN apt-get update -y && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /tmp/* \
    && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install soundfile

ADD *.py *.yml ./

ADD videos/ ./videos

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "preprocessor.vggish.yml", "--read_only"]