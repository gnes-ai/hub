FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    vim \
    git \
    wget \
    curl \
    cmake \
    unzip \
    gcc \
    g++ \
    automake \
    software-properties-common \
    ffmpeg \
    && apt-get clean

RUN pip install -U git+https://github.com/gnes-ai/gnes.git@remove-shot-detector --no-cache-dir --compile

ADD . ./

RUN python -m unittest test_shot_detector.py -v

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "detector.edge.yml", "--py_path", "shot_detector.py", "--read_only"]
