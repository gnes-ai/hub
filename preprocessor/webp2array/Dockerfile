FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        cmake git && \
    apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install -U cffi conan pillow peakutils --no-cache-dir --compile && \
    rm -rf /tmp/*

RUN conan install libwebp/1.0.3@bincrafters/stable --build libwebp

RUN pip install -U git+https://github.com/numb3r3/pywebp.git@master --compile && \
    rm -rf /tmp/*

ADD webp2array.py webp2array.yml ./

ENTRYPOINT ["gnes", "preprocess", "--yaml_path", "webp2array.yml", "--py_path", "webp2array.py", "--read_only"]