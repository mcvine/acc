FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
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

# install and setup mamba
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh -O ~/mambaforge.sh \
    && /bin/bash ~/mambaforge.sh -b -p /opt/conda \
    && rm ~/mambaforge.sh \
    && /opt/conda/bin/mamba clean -qay \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && ln -s /opt/conda/etc/profile.d/mamba.sh /etc/profile.d/mamba.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo ". /opt/conda/etc/profile.d/mamba.sh" >> ~/.bashrc \
    && echo "mamba activate base" >> ~/.bashrc
