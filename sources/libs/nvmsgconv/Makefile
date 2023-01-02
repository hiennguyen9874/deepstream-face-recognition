###############################################################################
#
# Copyright (c) 2018-2022 NVIDIA CORPORATION.  All Rights Reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
###############################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CC:= g++

PKGS:= glib-2.0 gobject-2.0 json-glib-1.0 uuid

NVDS_VERSION:=6.1

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -Wall -std=c++11 -shared -fPIC

CFLAGS+= -fPIC -DDS_VERSION=\"6.1.1\" \
  -I /usr/local/cuda-$(CUDA_VER)/include \
  -I../../includes \
  -I./deepstream_schema \
  -DNDEBUG

CFLAGS+= -I../../includes -I./deepstream_schema

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -shared -Wl,-no-undefined \
   -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl \
	 -lcudart -lpthread -I/usr/include/gstreamer-1.0 \
   -I/usr/include/glib-2.0 \
   -I/usr/lib/x86_64-linux-gnu/glib-2.0/include \
	 -lnppc -lnppig -lnpps -lnppicc -lnppidei \
   -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform\
	 -Wl,-rpath,$(LIB_INSTALL_DIR)

LIBS+= -lyaml-cpp

SRCFILES:= nvmsgconv.cpp  \
  deepstream_schema/eventmsg_payload.cpp \
  deepstream_schema/dsmeta_payload.cpp \
  deepstream_schema/deepstream_schema.cpp
TARGET_LIB:= libnvds_msgconv.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) ../../../lib

clean:
	rm -rf $(TARGET_LIB)
