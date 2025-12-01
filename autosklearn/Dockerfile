# força arquitetura x86_64
FROM --platform=linux/amd64 ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# instalar dependências
RUN apt-get update -y && \
    apt-get install -y \
        python3.9 python3.9-dev python3.9-distutils \
        wget curl git build-essential swig \
        libatlas-base-dev liblapack-dev libblas-dev \
        libglib2.0-0 libsm6 libxrender1 libxext6 && \
    apt-get clean

# instalar pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# atualizar ferramentas
RUN python3.9 -m pip install --upgrade pip setuptools wheel "numpy<2" cython==0.29.36

# instalar versão estável do auto-sklearn
RUN python3.9 -m pip install auto-sklearn==0.15.0

WORKDIR /workspace
COPY . /workspace

CMD ["/bin/bash"]
