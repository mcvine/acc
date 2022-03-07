#!/usr/bin/bash
export EXE=sgm
time mpirun -n 1 ./$EXE -n1e9 -d out
