#!/usr/bin/bash
export SRC=sgm.c
export EXE=sgm
export OMPI_CC=nvc
mpicc -acc -ta:tesla,managed -Minfo:accel -DUSE_MPI -DOPENACC $SRC -l mpi -lm -o $EXE
