################################################################################
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

CUDA_VER?=11.4
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CC:= g++

CFLAGS:= -Wall -std=c++11

LIB_INSTALL_DIR?=../../../lib

CFLAGS+= -shared -fPIC -fopenmp

CFLAGS+= -I../../includes \
         -I/usr/local/cuda-$(CUDA_VER)/include -I/usr/local/include/faiss

LIBS:= -lnvinfer -lnvparsers
LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas
LIBS+= -L/usr/local/lib/ -lfaiss /usr/local/lib/libfaiss.so

LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsinfer_custombboxparser.cpp \
		nvdsinfer_customfacedetectionparser.cpp \
		nvdsinfer_customfacerecognition.cpp \
		nvdsinfer_customfacerecognition_gpu.cpp \
		nvdsinfer_custombboxlandmarkparser.cpp

TARGET_LIB:= libnvds_infercustomparser.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(TARGET_LIB)
	cp $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB)
