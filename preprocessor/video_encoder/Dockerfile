FROM gnes/gnes:latest-develop

LABEL maintainer="team@gnes.ai"

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        ffmpeg && \
    apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

ADD *.py *.yml *.npy ./

RUN python -m unittest test_encoder.py -v

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "mp4_encoder.yml", "--py_path", "video_encoder.py"]