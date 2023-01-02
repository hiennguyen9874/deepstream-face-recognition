#!/usr/bin/env bash

export PWD=`pwd`
export DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

TARGET_DEVICE=$(gcc -dumpmachine | cut -f1 -d -)
if [ "$TARGET_DEVICE" == "aarch64" ]; then
export CUDA_VER=11.4
else
export CUDA_VER=11.7
fi

cd $DIR

echo "Building deepstream-app"
cd apps/sample_apps/deepstream-app && make clean -j4 && cd $DIR

echo "Building gst-nvinfer"
cd gst-plugins/gst-nvinfer && make clean -j4 && cd $DIR

echo "Building nvdsinfer_customparser"
cd libs/nvdsinfer_customparser && make clean -j4 && cd $DIR

echo "Building nvmsgconv"
cd libs/nvmsgconv && make clean -j4 && cd $DIR

echo "Building nvdsinfer"
cd libs/nvdsinfer && make clean -j4 && cd $DIR

echo "Build done!"

cd $PWD
