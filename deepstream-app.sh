# export GST_DEBUG=3
# export NVDSINFER_LOG_LEVEL=10
# export USE_NEW_NVSTREAMMUX=yes
export GST_DEBUG_DUMP_DOT_DIR="logs/"
# export NVDS_ENABLE_LATENCY_MEASUREMENT=1
# export NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1

# export LD_DEBUG=libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:/lib/x86_64-linux-gnu/:/lib/x86_64-linux-gnu/:/lib64/

sudo rm -rf **/*.so
# sudo bash sources/uninstall.sh
sudo bash sources/default.sh
sudo bash sources/install.sh
rm -rf logs/*
./bin/deepstream-app -c samples/configs/deepstream_app.txt
python3 export_svg.py
sudo bash sources/default.sh
