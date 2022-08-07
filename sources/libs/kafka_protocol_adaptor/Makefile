################################################################################
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

# this Makefile is to be used to build the kafka_proto protocol adaptor .so
CXX:=g++

PKGS:= glib-2.0 

SRCS:=  nvds_kafka_proto.cpp kafka_client.cpp json_helper.cpp nvds_utils.cpp
TARGET_LIB:= libnvds_kafka_proto.so 

CFLAGS:= -fPIC -Wall

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))
LDFLAGS:= -shared

DS_INC:= ../../includes
RDKAFKA_INC:=/usr/local/include/librdkafka

INC_PATHS:= -I $(DS_INC) -I $(RDKAFKA_INC)
CFLAGS+= $(INC_PATHS)

LIBS+= -L../../lib -lrdkafka -ljansson -lnvds_logger -lpthread -lcrypto
LDFLAGS+= -shared

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCS)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

clean:
	rm -rf $(TARGET_LIB)

