#!/bin/bash

make CXX=g++ CXXFLAGS= LDFLAGS= SUFFIX=-gcc

LBT_DEFAULT_LIBS=/usr/lib/libopenblas_64.so \
LBT_STRICT=1 \
LBT_VERBOSE=1 \
OMP_PROC_BIND=true \
OPENBLAS_MAIN_FREE=1 \
OPENBLAS_NUM_THREADS=1 \
exec ./lbtbench
