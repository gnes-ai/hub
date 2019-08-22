FROM pytorch/pytorch

LABEL maintainer="team@gnes.ai"

RUN pip install gnes[flair]

ADD *.py *.yml ./

# [Optional] run a simple unit test
# you probably want to comment out this line in the CICD pipeline,
# as your CI runner for "docker build" may not have enough memory
RUN python -m unittest test_flair.py -v

ENTRYPOINT ["gnes", "encode", "--yaml_path", "flair.yml", "--read_only"]