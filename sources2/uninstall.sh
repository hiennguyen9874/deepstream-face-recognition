#!/usr/bin/env bash

export PWD=`pwd`
export DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $DIR

echo "Cleaning deepstream-app"
cd apps/sample_apps/deepstream-app && make clean && cd $DIR

echo "Cleaning gst-dsimage-save"
cd gst-plugins/gst-dsimage-save && make clean && cd $DIR

# echo "Cleaning gst-dspostprocessing"
# cd gst-plugins/gst-dspostprocessing-kmeans && make clean && cd $DIR

echo "Cleaning gst-dspostprocessing"
cd gst-plugins/gst-dspostprocessing-without-kmeans && make clean && cd $DIR

echo "Cleaning nvdsinfer"
cd libs/nvdsinfer && make clean && cd $DIR

echo "Cleaning nvdsinfer custom parser"
cd libs/nvdsinfer_customparser && make clean && cd $DIR

echo "Cleaning nvmsgconv"
cd libs/nvmsgconv && make clean && cd $DIR

echo "Build done!"

cd $PWD
