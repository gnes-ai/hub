FROM gnes/gnes:latest-alpine

LABEL maintainer="team@gnes.ai"

RUN apk add --no-cache \
            build-base g++ gfortran file binutils \
            musl-dev python3-dev py-pgen cython openblas-dev && \
    pip install annoy --no-cache-dir --compile && \
    find /usr/lib/python3.7/ -name 'tests' -exec rm -r '{}' + && \
    find /usr/lib/python3.7/site-packages/ -name '*.so' -print -exec sh -c 'file "{}" | grep -q "not stripped" && strip -s "{}"' \; && \
    rm -rf /tmp/*

WORKDIR /workspace/

ADD *.py *.yml ./

ENTRYPOINT ["gnes", "index", "--yaml_path", "annoy.yml", "--py_path", "_annoy.py"]