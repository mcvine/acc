#!/usr/bin/bash
export DIMG=nvcr.io/nvidia/nvhpc:20.11-devel-cuda_multi-ubuntu20.04
docker run --gpus all --rm -it --user $(id -u):$(id -g) -v $(pwd):/work -w /work $DIMG ./run.sh
