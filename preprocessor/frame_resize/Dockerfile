FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN pip install Pillow

ADD resize.py resize.yml ./

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "resize.yml", "--py_path", "resize.py", "--read_only"]