FROM tensorflow/tensorflow:latest-gpu-py3

LABEL maintainer="team@gnes.ai"

RUN pip install -U gnes --no-cache-dir --compile

ADD *.py *.yml ./

ADD vggish_cores/ ./vggish_cores

RUN python -m unittest test_vggish.py -v

ENTRYPOINT ["gnes", "encode", "--yaml_path", "encoder.vggish.yml", "--py_path", "vggish.py", "--read_only"]