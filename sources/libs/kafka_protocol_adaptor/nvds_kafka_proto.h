/*
 * Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#ifndef __NVDS_KAFKA_PROTO_H__
#define __NVDS_KAFKA_PROTO_H__

#define NVDS_MSGAPI_VERSION "2.0"
#define NVDS_MSGAPI_PROTOCOL "KAFKA"

#define CONFIG_GROUP_MSG_BROKER_RDKAFKA_CFG "proto-cfg"
#define CONFIG_GROUP_MSG_BROKER_RDKAFKA_PRODUCER_CFG "producer-proto-cfg"
#define CONFIG_GROUP_MSG_BROKER_RDKAFKA_CONSUMER_CFG "consumer-proto-cfg"
#define CONFIG_GROUP_MSG_BROKER_PARTITION_KEY "partition-key"
#define CONFIG_GROUP_MSG_BROKER_CONSUMER_GROUP "consumer-group-id"
#define CONFIG_GROUP_MSG_BROKER_RDKAFKA_SHARE_CONNECTION "share-connection"
#define DEFAULT_KAFKA_CONSUMER_GROUP "test-consumer-group"
#define DEFAULT_PARTITION_NAME "sensor.id"

#endif
