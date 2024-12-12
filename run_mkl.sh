#!/bin/bash

export LD_LIBRARY_PATH="$(python3 -c 'import os, sys; p = sys.argv[2].split(chr(58)); print(chr(58).join(p + [i for i in os.getenv(sys.argv[1], str()).split(chr(58)) if i and i not in p]))' LD_LIBRARY_PATH /opt/intel/oneapi/compiler/latest/lib)"

make CXX=/opt/intel/oneapi/compiler/latest/bin/icpx CXXFLAGS= LDFLAGS= SUFFIX=-icc

LBT_DEFAULT_LIBS='/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so!64' \
LBT_STRICT=1 \
LBT_VERBOSE=1 \
MKL_INTERFACE_LAYER=ILP64 \
MKL_THREADING_LAYER=SEQUENTIAL \
OMP_PROC_BIND=true \
exec build/lbtbench-icc
