FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN pip install -U webp pillow --no-cache-dir --compile && \
     rm -rf /tmp/*

ADD webp2array.py webp2array.yml ./

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "webp2array.yml", "--py_path", "webp2array.py", "--read_only"]