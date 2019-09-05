FROM gnes/hub-indexer:latest-faiss-cpu

LABEL maintainer="team@gnes.ai"

RUN /opt/conda/bin/pip install -U sklearn --no-cache-dir --compile

ADD pca.py test_pca.py pca.yml pcalocal.yml ./

RUN /opt/conda/bin/python -m unittest test_pca.py -v

ENTRYPOINT ["/opt/conda/bin/gnes", "encode", "--yaml_path", "pca.yml", "--py_path", "pca.py", "--read_only"]