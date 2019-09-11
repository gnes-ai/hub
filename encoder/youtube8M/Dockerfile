FROM tensorflow/tensorflow:latest-gpu-py3

LABEL maintainer="team@gnes.ai"

RUN pip install -U gnes --no-cache-dir --compile

RUN pip install pillow

ADD *.py *.yml ./ 

ADD yt8m_feature_extractor_cores/ ./yt8m_feature_extractor_cores

ENTRYPOINT ["gnes", "encode", "--yaml_path", "encoder.yt8m.yml", "--py_path", "yt8m_feature_extractor.py", "yt8m_model.py", "--read_only"]