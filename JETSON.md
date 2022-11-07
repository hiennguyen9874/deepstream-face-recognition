# Default runtime
https://github.com/dusty-nv/jetson-containers#docker-default-runtime

# Install custom tensorRT
- Build tensorrt
    ```
    ARG TRT_OSS_CHECKOUT_TAG=release/8.2
    ARG TENSORRT_REPO=https://github.com/nvidia/TensorRT
    WORKDIR /tmp
    RUN git clone -b $TRT_OSS_CHECKOUT_TAG $TENSORRT_REPO
    WORKDIR /tmp/TensorRT
    ENV TRT_SOURCE=/tmp/TensorRT
    WORKDIR $TRT_SOURCE
    RUN git submodule update --init --recursive
    RUN mkdir -p build
    WORKDIR /tmp/TensorRT/build
    RUN ls /usr/src/tensorrt
    RUN /cmake/bin/cmake .. -DGPU_ARCHS="62" -DTENSORRT_ROOT=/usr/src/tensorrt -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.2/bin/nvcc -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include -DTRT_LIB_DIR=/usr/lib/aarch64-linux-gnu/ -DCMAKE_C_COMPILER=/usr/bin/gcc -DTRT_BIN_DIR=`pwd`/out
    RUN make nvinfer_plugin -j$(nproc)
    RUN cp $(find /tmp/TensorRT/build -name "libnvinfer_plugin.so.8.*" -print -quit) $(find /usr/lib/aarch64-linux-gnu/ -name "libnvinfer_plugin.so.8.*" -print -quit)
    RUN ldconfi
    COPY ./libnvinfer_plugin.so.8.2.3 /tmp
    RUN echo $(find /usr/lib/aarch64-linux-gnu/ -name "libnvinfer_plugin.so.8.*" -print -quit)
    RUN cp /tmp/libnvinfer_plugin.so.8.2.3 $(find /usr/lib/aarch64-linux-gnu/ -name "libnvinfer_plugin.so.8.*" -print -quit)
    RUN ldconfig
    ```
- cp ./libnvinfer_plugin.so.8.2.3 $(find /usr/lib/aarch64-linux-gnu/ -name 'libnvinfer_plugin.so.8.*' -print -quit)
- ldconfig

# Docker
- `docker build -t hiennguyen9874/deepstream-face-recognition:jetson-deepstream-6.0.1 -f Dockerfile.jetson .`
- `docker push hiennguyen9874/deepstream-face-recognition:jetson-deepstream-6.0.1`

# Install Opencv
Inside docker or outside docker
https://forums.developer.nvidia.com/t/jetson-docker-image-opencv/164792

# Install Faiss
Inside docker

# TODO
- Opencv
- Faiss
- Mount opencv from host into docker using nvidia-container-runtime: https://github.com/dusty-nv/jetson-containers/issues/5
- Flip video

# TensorRT custom error
[11/05/2022-02:13:31] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 2243 detected for tactic 4.
Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
[11/05/2022-02:13:32] [W] [TRT] Tactic Device request: 2218MB Available: 2189MB. Device memory is insufficient to use tactic.
[11/05/2022-02:13:32] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 2218 detected for tactic 4.
Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
[11/05/2022-02:14:30] [W] [TRT] Tactic Device request: 4461MB Available: 2343MB. Device memory is insufficient to use tactic.
[11/05/2022-02:14:30] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 4461 detected for tactic 4.
Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
[11/05/2022-02:14:31] [W] [TRT] Tactic Device request: 4423MB Available: 2343MB. Device memory is insufficient to use tactic.
[11/05/2022-02:14:31] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 4423 detected for tactic 4.
Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
[11/05/2022-02:14:49] [W] [TRT] Tactic Device request: 4327MB Available: 2386MB. Device memory is insufficient to use tactic.
[11/05/2022-02:14:49] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 4327 detected for tactic 4.
Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
[11/05/2022-02:14:50] [W] [TRT] Tactic Device request: 4308MB Available: 2387MB. Device memory is insufficient to use tactic.
[11/05/2022-02:14:50] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 4308 detected for tactic 4.
Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
[11/05/2022-02:15:13] [W] [TRT] Tactic Device request: 4309MB Available: 2389MB. Device memory is insufficient to use tactic.
[11/05/2022-02:15:13] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 4309 detected for tactic 4.
Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
[11/05/2022-02:15:14] [W] [TRT] Tactic Device request: 4299MB Available: 2388MB. Device memory is insufficient to use tactic.
[11/05/2022-02:15:14] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 4299 detected for tactic 4.
Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().