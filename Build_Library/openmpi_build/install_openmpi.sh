#!/bin/bash

# ==================
# Install openmpi library
# Tested successfully on Amazon-Linux AMI
#
# Jiawei Zhuang 2017/4
# ==================

# ==================
# make a new directory if not exist
# ==================
mkdir -p $HOME/lib
cd $HOME/lib

# ==================
# openmpi build (make install takes many minutes)
# Some of the "make check "tests might fail, 
# but it doesn't affect basic MPI functionality
# ==================

wget https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.0.tar.gz
tar xvf openmpi-2.1.0.tar.gz 
cd openmpi-2.1.0
./configure --prefix=/usr/local/openmpi
make check
sudo make install
