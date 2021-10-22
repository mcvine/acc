FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    apt-utils \
    bzip2 \
    ca-certificates \
    curl \
    freeglut3-dev \
    git \
    libglu1-mesa \
    procps \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# install and setup conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda clean -tipsy \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc
