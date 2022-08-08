#!/usr/bin/env bash

export PWD=`pwd`
export DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export CUDA_VER=11.4

cd $DIR

echo "Cleaning gst-dsexample"
cd gst-plugins/gst-dsexample && make && make clean && cd $DIR

echo "Cleaning gst-nvdsosd"
cd gst-plugins/gst-nvdsosd && make && make clean && cd $DIR

# echo "Cleaning gst-nvdsspeech"
# cd gst-plugins/gst-nvdsspeech && make && make clean && cd $DIR

echo "Cleaning gst-nvdsvideotemplate"
cd gst-plugins/gst-nvdsvideotemplate && make && make clean && cd $DIR

echo "Cleaning gst-nvmsgbroker"
cd gst-plugins/gst-nvmsgbroker && make && make clean && cd $DIR

echo "Cleaning gst-nvdsaudiotemplate"
cd gst-plugins/gst-nvdsaudiotemplate && make && make clean && cd $DIR

echo "Cleaning gst-nvdspreprocess"
cd gst-plugins/gst-nvdspreprocess && make && make clean && cd $DIR

# echo "Cleaning gst-nvdstexttospeech"
# cd gst-plugins/gst-nvdstexttospeech && make && make clean && cd $DIR

echo "Cleaning gst-nvinfer"
cd gst-plugins/gst-nvinfer && make && make clean && cd $DIR

echo "Cleaning gst-nvmsgconv"
cd gst-plugins/gst-nvmsgconv && make && make clean && cd $DIR

# echo "Cleaning amqp_protocol_adaptor"
# cd libs/amqp_protocol_adaptor && make && make clean && cd $DIR

# echo "Cleaning kafka_protocol_adaptor"
# cd libs/kafka_protocol_adaptor && make && make clean && cd $DIR

echo "Cleaning nvdsinfer_customparser"
cd libs/nvdsinfer_customparser && make && make clean && cd $DIR

echo "Cleaning nvmsgconv"
cd libs/nvmsgconv && make && make clean && cd $DIR

# echo "Cleaning redis_protocol_adaptor"
# cd libs/redis_protocol_adaptor && make && make clean && cd $DIR

# echo "Cleaning azure_protocol_adaptor"
# cd libs/azure_protocol_adaptor && make && make clean && cd $DIR

echo "Cleaning nvdsinfer"
cd libs/nvdsinfer && make && make clean && cd $DIR

# echo "Cleaning nvmsgbroker"
# cd libs/nvmsgbroker && make && make clean && cd $DIR

echo "Cleaning nvmsgconv_audio"
cd libs/nvmsgconv_audio && make && make clean && cd $DIR

echo "Cleaning deepstream-app"
cd apps/sample_apps/deepstream-app && make && make clean && cd $DIR

echo "Build done!"

cd $PWD






