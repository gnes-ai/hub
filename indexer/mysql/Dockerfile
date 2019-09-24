FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN pip install PyMySQL

ADD mysql_idx.py mysql_idx.yml ./

ENTRYPOINT ["gnes", "index", "--py_path", "mysql_idx.py", "--yaml_path", "mysql_idx.yml", "--read_only"]