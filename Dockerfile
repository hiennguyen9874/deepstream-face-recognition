ARG BASE_CONTAINER=nvcr.io/nvidia/deepstream:6.0.1-devel
FROM $BASE_CONTAINER as base

# Enviroment
ENV SWIG_PATH=/usr/swig/bin
ENV PATH=$SWIG_PATH:$PATH
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

ARG DISTRO=ubuntu1804
ARG ARCH=x86_64

RUN apt-key del 7fa2af80 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/* \
    && sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libatlas-base-dev libatlas3-base \
    clang-8 \
    libopenblas-dev \
    libpcre2-dev \
    flex bison \
    libglib2.0-dev \
    libjson-glib-dev \
    uuid-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt autoremove \
    && apt-get clean

# Cmake
WORKDIR /tmp
# RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.4/cmake-3.19.4.tar.gz \
#     && tar xvf cmake-3.19.4.tar.gz \
#     && rm cmake-3.19.4.tar.gz \
#     && cd /tmp/cmake-3.19.4/ \
#     && mkdir /cmake \
#     && ./configure --prefix=/cmake \
#     && make -j$(nproc) \
#     && make install \
#     && cd /tmp \
#     && rm -rf /tmp/cmake-3.19.4/
RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.5/cmake-3.19.5-Linux-x86_64.tar.gz \
    && tar -zxvf cmake-3.19.5-Linux-x86_64.tar.gz \
    && rm cmake-3.19.5-Linux-x86_64.tar.gz \
    && cd /tmp/cmake-3.19.5-Linux-x86_64/ \
    && cp -rf bin/ doc/ share/ /usr/local/ \
    && cp -rf man/* /usr/local/man \
    && sync \
    && cmake --version \
    && cd /tmp \
    && rm -rf /tmp/cmake-3.19.5-Linux-x86_64/

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools Cython wheel \
    && python3 -m pip install --no-cache-dir numpy

FROM base as builder

# Build tensorRT
ARG TRT_OSS_CHECKOUT_TAG=release/8.2
ARG DGPU_ARCHS=75
ARG TENSORRT_REPO=https://github.com/hiennguyen9874/TensorRT

WORKDIR /tmp
RUN git clone -b $TRT_OSS_CHECKOUT_TAG $TENSORRT_REPO \
    && export TRT_SOURCE=/tmp/TensorRT \
    && cd /tmp/TensorRT \
    && git submodule update --init --recursive \
    && mkdir -p build \
    && cd /tmp/TensorRT/build \
    && cmake .. -DGPU_ARCHS=$DGPU_ARCHS \
    -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu/ \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DTRT_BIN_DIR=`pwd`/out \
    && make nvinfer_plugin -j$(nproc) \
    && cp $(find /tmp/TensorRT/build -name "libnvinfer_plugin.so.8.*" -print -quit) \
    $(find /usr/lib/x86_64-linux-gnu/ -name "libnvinfer_plugin.so.8.*" -print -quit) \
    && ldconfig \
    && cd /tmp \
    && rm -rf /tmp/TensorRT

WORKDIR /tmp
RUN wget https://prdownloads.sourceforge.net/swig/swig-4.1.0.tar.gz \
    && tar -xzvf swig-4.1.0.tar.gz \
    && rm swig-4.1.0.tar.gz \
    && cd /tmp/swig-4.1.0 \
    && ./configure --prefix=/usr/swig \
    && make \
    && make install \
    && cd /tmp \
    && rm -rf /tmp/swig-4.1.0
# /usr/swig/
ENV SWIG_PATH=/usr/swig/bin
ENV PATH=$SWIG_PATH:$PATH

WORKDIR /tmp
RUN git clone https://github.com/facebookresearch/faiss.git \
    && cd /tmp/faiss \
    && mkdir build \
    && cmake -B build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
    -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    -DPython_EXECUTABLE=$(which python3) \
    -DFAISS_OPT_LEVEL=generic \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON . \
    && make -C build -j faiss \
    && make -C build -j swigfaiss \
    && make -C build install \
    && cd /tmp \
    && mkdir /tmp/faiss-python \
    && cp -r /tmp/faiss/build/faiss /tmp/faiss-python \
    && rm -rf /tmp/faiss
# /usr/local/include/faiss, /usr/local/share/faiss, /usr/local/lib/libfaiss.so

FROM base

COPY --from=builder /usr/lib/x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/swig /usr/swig
COPY --from=builder /usr/local/include/faiss /usr/local/include/faiss
COPY --from=builder /usr/local/share/faiss /usr/local/share/faiss
COPY --from=builder /usr/local/lib/libfaiss.so /usr/local/lib/libfaiss.so

COPY --from=builder /tmp/faiss-python /tmp/faiss-python
WORKDIR /tmp
RUN cd /tmp/faiss-python/faiss/python \
    && python3 setup.py install \
    && cp /tmp/faiss-python/faiss/python/*.so /usr/local/lib/ \
    && cd /tmp \
    && rm -rf /tmp/faiss-python

RUN python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    aiohttp numpy scipy \
    scikit-image matplotlib \
    pycuda six

RUN apt-get update && apt-get install -yq --no-install-recommends ffmpeg libsm6 libxext6 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir opencv-python

USER root

WORKDIR /app
COPY ./ /app

CMD ["./bin/deepstream-app", "-c", "samples/configs/deepstream_app.txt"]
