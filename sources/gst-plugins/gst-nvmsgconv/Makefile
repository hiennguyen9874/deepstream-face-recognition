################################################################################
# Copyright (c) 2018-2021 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#################################################################################

CXX:= gcc
SRCS:= gstnvmsgconv.c
INCS:= gstnvmsgconv.h
LIB:=libnvdsgst_msgconv.so

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=6.0

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS+= -fPIC -DDS_VERSION=\"6.0.1\" \
	 -I../../includes \
	 -I../../libs/nvmsgconv/

LIBS := -shared -Wl,-no-undefined
LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvds_meta -lnvds_msgconv -ldl \
       -Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.c=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.c $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
