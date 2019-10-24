FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN pip install PyMySQL

ADD mysql.py mysql.yml ./

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "mysql.yml", "--py_path", "mysql.py", "--read_only"]