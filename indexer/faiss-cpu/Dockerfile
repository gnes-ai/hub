FROM continuumio/anaconda3

LABEL maintainer="team@gnes.ai"

RUN apt-get update && apt-get install -y build-essential && \
    /opt/conda/bin/conda install faiss-cpu -c pytorch && \
    /opt/conda/bin/pip install gnes --no-cache-dir --compile && \
    rm -rf /var/lib/apt/lists/*

ADD *.py *.yml ./

ENTRYPOINT ["/opt/conda/bin/gnes", "index", "--yaml_path", "faiss.yml", "--py_path", "faiss.py", "--read_only"]