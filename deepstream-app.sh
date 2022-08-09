# export GST_DEBUG=3
# export NVDSINFER_LOG_LEVEL=10
# export USE_NEW_NVSTREAMMUX=yes
export GST_DEBUG_DUMP_DOT_DIR="logs/"
# export NVDS_ENABLE_LATENCY_MEASUREMENT=1
# export NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/lib:/opt/nvidia/deepstream/deepstream/lib/gst-plugins
sudo bash sources/default.sh
sudo bash sources/install.sh
rm -rf logs/*
/opt/nvidia/deepstream/deepstream-6.0/bin/deepstream-app -c samples/configs/deepstream_app.txt
python3 export_svg.py
sudo bash sources/default.sh
