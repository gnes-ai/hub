FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

ADD file-sys.py file-sys.yml ./

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "file-sys.yml", "--py_path", "file-sys.py", "--read_only"]