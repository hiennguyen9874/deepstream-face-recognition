#!/usr/bin/env bash

export PWD=`pwd`
export DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# export CUDA_VER=11.4

cd $DIR

echo "Building deepstream-app"
cd apps/sample_apps/deepstream-app && make install -j4 && cd $DIR

echo "Building gst-dsexample"
cd gst-plugins/gst-dsexample && make install -j4 && cd $DIR

echo "Building gst-nvinfer"
cd gst-plugins/gst-nvinfer && make install -j4 && cd $DIR

echo "Building nvdsinfer_customparser"
cd libs/nvdsinfer_customparser && make install -j4 && cd $DIR

echo "Building nvmsgconv"
cd libs/nvmsgconv && make install -j4 && cd $DIR

echo "Building nvdsinfer"
cd libs/nvdsinfer && make install -j4 && cd $DIR

echo "Build done!"

cd $PWD
