FROM continuumio/anaconda3

LABEL maintainer="team@gnes.ai"

RUN apt-get update && apt-get install -y build-essential && \
    /opt/conda/bin/conda install faiss-cpu -c pytorch && \
    /opt/conda/bin/pip install git+https://github.com/gnes-ai/gnes.git@index_dev --no-cache-dir --compile && \
    rm -rf /var/lib/apt/lists/*

ADD *.yml ./

ENTRYPOINT ["/opt/conda/bin/gnes", "index", "--yaml_path", "faiss.yml"]