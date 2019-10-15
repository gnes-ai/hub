FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN pip install Pillow

ADD frame_select.py frame_select.yml ./

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "frame_select.yml", "--py_path", "frame_select.py", "--read_only"]