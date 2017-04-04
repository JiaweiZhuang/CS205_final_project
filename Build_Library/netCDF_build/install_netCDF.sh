#!/bin/bash

# ==================
# Install netCDF-C library
# Tested successfully on Amazon-Linux AMI
#
# Jiawei Zhuang 2017/4
# ==================

# ==================
# Note:
# Use the zlib,HDF5,NetCDF4 versions specified in 
# https://github.com/amznlabs/amazon-dsstne/blob/master/docs/getting_started/setup.md#openmpi-setup
# but added more --prefix and include options according to
# http://www.unidata.ucar.edu/software/netcdf/docs/getting_and_building_netcdf.html#build_default
#
# The older version(netcdf 4.1.3) seems much easier to install than the lastest version (netcdf 4.4.1) 
# ==================

# ==================
# for C compiler if not installed yet
# ==================
#sudo yum install gcc
#sudo yum install gcc-c++
#CC=gcc
#CXX=g++

# ==================
# make a new directory if not exist
# ==================
mkdir -p $HOME/lib
cd $HOME/lib

# ==================
# for zlib
# ==================
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/zlib-1.2.8.tar.gz
tar xvf zlib-1.2.8.tar.gz
cd zlib-1.2.8

# Build and install zlib
ZDIR=/usr/local
./configure --prefix=${ZDIR}
make check
sudo make install

cd ..

# ==================
# for HDF5 
# The "make check" step takes 10~20 minutes
# Some of the tests might fail, but doesn't affect netCDF functionality
# ==================
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/hdf5-1.8.12.tar.gz
tar xvfz hdf5-1.8.12.tar.gz
cd hdf5-1.8.12

# Build and install HDF5
H5DIR=/usr/local
./configure --with-zlib=${ZDIR} --prefix=${H5DIR}
make check
sudo make install

cd ..

# ==================
# for m4 if necessary
# (https://geeksww.com/tutorials/libraries/m4/installation/installing_m4_macro_processor_ubuntu_linux.php)
# ==================

# ==================
# for netCDF4
# The "make check" step takes 5~10 minutes
# ==================

wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.1.3.tar.gz
tar xvf netcdf-4.1.3.tar.gz
cd netcdf-4.1.3

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${H5DIR}/lib

# Build and install netCDF-4. We don't need Fortran support (no gfortran installed)
NCDIR=/usr/local
CPPFLAGS=-I${H5DIR}/include LDFLAGS=-L${H5DIR}/lib ./configure --prefix=${NCDIR} --disable-fortran
make check # will fail fortran check without "--disable-fortran" in the configure step
sudo make install

# show the configure details
nc-config --all
