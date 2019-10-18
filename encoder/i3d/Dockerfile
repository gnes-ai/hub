FROM tensorflow/tensorflow:1.14.0-gpu-py3

LABEL maintainer="team@gnes.ai"

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        git && \
    apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install dm-sonnet==1.23

RUN pip install git+https://github.com/gnes-ai/gnes.git@master

RUN apt-get update && apt-get install -y vim && \
    rm -rf /var/lib/apt/lists/*

ADD checkpoints ./checkpoints

ADD i3d_cores ./i3d_cores

ADD test_i3d.py i3d_encoder.py i3d_encoder.yml ./

ENTRYPOINT ["gnes", "encode", "--yaml_path", "i3d_encoder.yml", "--py_path", "./i3d_cores/i3d.py", "i3d_encoder.py", "--read_only"]
