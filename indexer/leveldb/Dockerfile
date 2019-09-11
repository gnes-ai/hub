FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN pip install plyvel>=1.0.5 --no-cache-dir --compile

ADD leveldb.yml ./

ENTRYPOINT ["gnes", "index", "--yaml_path", "leveldb.yml"]