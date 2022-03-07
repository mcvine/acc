#!/bin/bash

rm -rf out sgm.c sgm
./instr2c.sh
./compile_on_docker.sh
./run_on_docker.sh
