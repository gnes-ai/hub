FROM gnes/gnes:latest-alpine

LABEL maintainer="team@gnes.ai"

ADD *.py *.yml ./

RUN python -m unittest test_pipeline.py -v

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "pipline.yml", "--py_path", "mypreprocessor1.py", "mypreprocessor2.py", "--read_only"]