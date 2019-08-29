FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /tmp/* && rm -rf /gnes \
    && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*


WORKDIR /shot_detector

ADD *.py *.yml requirements.txt ./
ADD videos ./videos

RUN pip install -U -r requirements.txt --no-cache-dir --compile

RUN python -m unittest test_shot_detector.py -v

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "detector.edge.yml", "--py_path", "shot_detector.py", "--read_only"]
