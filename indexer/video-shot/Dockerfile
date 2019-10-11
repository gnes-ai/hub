FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN pip install -U webp pillow --no-cache-dir --compile && \
    rm -rf /tmp/*

ADD *.py *.yml *.npy ./

RUN python -m unittest test_*.py -v

ENTRYPOINT ["gnes", "index", "--py_path", "video_shot_indexer.py"]