#!/usr/bin/env bash

export PWD=`pwd`
export DIR='/opt/nvidia/deepstream/deepstream/sources/'
export CUDA_VER=11.4

cd $DIR

echo "Building deepstream-app"
cd apps/sample_apps/deepstream-app && make install -j4 && cd $DIR

echo "Building gst-dsexample"
cd gst-plugins/gst-dsexample && make install -j4 && cd $DIR

echo "Building gst-nvdsosd"
cd gst-plugins/gst-nvdsosd && make install -j4 && cd $DIR

# echo "Building gst-nvdsspeech"
# cd gst-plugins/gst-nvdsspeech && make install -j4 && cd $DIR

echo "Building gst-nvdsvideotemplate"
cd gst-plugins/gst-nvdsvideotemplate && make install -j4 && cd $DIR

echo "Building gst-nvmsgbroker"
cd gst-plugins/gst-nvmsgbroker && make install -j4 && cd $DIR

echo "Building gst-nvdsaudiotemplate"
cd gst-plugins/gst-nvdsaudiotemplate && make install -j4 && cd $DIR

echo "Building gst-nvdspreprocess"
cd gst-plugins/gst-nvdspreprocess && make install -j4 && cd $DIR

# echo "Building gst-nvdstexttospeech"
# cd gst-plugins/gst-nvdstexttospeech && make install -j4 && cd $DIR

echo "Building gst-nvinfer"
cd gst-plugins/gst-nvinfer && make install -j4 && cd $DIR

echo "Building gst-nvmsgconv"
cd gst-plugins/gst-nvmsgconv && make install -j4 && cd $DIR

# echo "Building amqp_protocol_adaptor"
# cd libs/amqp_protocol_adaptor && make install -j4 && cd $DIR

# echo "Building kafka_protocol_adaptor"
# cd libs/kafka_protocol_adaptor && make install -j4 && cd $DIR

echo "Building nvdsinfer_customparser"
cd libs/nvdsinfer_customparser && make install -j4 && cd $DIR

echo "Building nvmsgconv"
cd libs/nvmsgconv && make install -j4 && cd $DIR

# echo "Building redis_protocol_adaptor"
# cd libs/redis_protocol_adaptor && make install -j4 && cd $DIR

# echo "Building azure_protocol_adaptor"
# cd libs/azure_protocol_adaptor && make install -j4 && cd $DIR

echo "Building nvdsinfer"
cd libs/nvdsinfer && make install -j4 && cd $DIR

# echo "Building nvmsgbroker"
# cd libs/nvmsgbroker && make install -j4 && cd $DIR

echo "Building nvmsgconv_audio"
cd libs/nvmsgconv_audio && make install -j4 && cd $DIR

echo "Build done!"

cd $PWD
