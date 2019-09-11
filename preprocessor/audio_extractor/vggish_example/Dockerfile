FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN apt-get update -y && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /tmp/* \
    && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install soundfile

ADD *.py *.yml ./

ADD vggish_example_helper/ ./vggish_example_helper

ADD videos/ ./videos

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "preprocessor.vggish.yml", "--py_path", "vggish_example.py", "--read_only"]