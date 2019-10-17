FROM gnes/gnes:latest-buster

LABEL maintainer="team@gnes.ai"

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        cmake git ffmpeg && \
    apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/* 

RUN pip install -U cffi conan pillow opencv-python>=4.0.0 peakutils --no-cache-dir --compile && \
    rm -rf /tmp/*

RUN conan install libwebp/1.0.3@bincrafters/stable --build libwebp

RUN pip install -U git+https://github.com/numb3r3/pywebp.git@master --compile && \
	rm -rf /tmp/*

ADD *.py ./
ADD test_yaml ./test_yaml
ADD test_data ./test_data

RUN python -m unittest test_*.py -v

ENTRYPOINT ["gnes"]
