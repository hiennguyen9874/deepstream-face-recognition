#!/usr/bin/env bash

export PWD=`pwd`
export DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $DIR

echo "Building deepstream-app"
cd apps/sample_apps/deepstream-app && make && make install && cd $DIR

echo "Building gst-dsimage-save"
cd gst-plugins/gst-dsimage-save && make && make install && cd $DIR

# echo "Building gst-dspostprocessing"
# cd gst-plugins/gst-dspostprocessing-kmeans && make && make install && cd $DIR

echo "Building gst-dspostprocessing"
cd gst-plugins/gst-dspostprocessing-without-kmeans && make && make install && cd $DIR

echo "Building nvdsinfer"
cd libs/nvdsinfer && make && make install && cd $DIR

echo "Building nvdsinfer custom parser"
cd libs/nvdsinfer_customparser && make && make install && cd $DIR

echo "Building nvmsgconv"
cd libs/nvmsgconv && make && make install && cd $DIR

echo "Build done!"

cd $PWD
