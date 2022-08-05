/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef NVDSPREPROCESS_PROPERTY_FILE_PARSER_H_
#define NVDSPREPROCESS_PROPERTY_FILE_PARSER_H_

#include <gst/gst.h>

#include "gstnvdspreprocess.h"

/**
 * This file describes the Macro defined for config file property parser.
 */

/** max string length */
#define _PATH_MAX 4096

#define NVDSPREPROCESS_PROPERTY "property"
#define NVDSPREPROCESS_PROPERTY_TARGET_IDS "target-unique-ids"
#define NVDSPREPROCESS_PROPERTY_ENABLE "enable"
#define NVDSPREPROCESS_PROPERTY_GPU_ID "gpu-id"
#define NVDSPREPROCESS_PROPERTY_PROCESSING_WIDTH "processing-width"
#define NVDSPREPROCESS_PROPERTY_PROCESSING_HEIGHT "processing-height"
#define NVDSPREPROCESS_PROPERTY_MAINTAIN_ASPECT_RATIO "maintain-aspect-ratio"
#define NVDSPREPROCESS_PROPERTY_TENSOR_BUF_POOL_SIZE "tensor-buf-pool-size"

#define NVDSPREPROCESS_PROPERTY_SCALING_BUF_POOL_SIZE "scaling-buf-pool-size"
#define NVDSPREPROCESS_PROPERTY_SCALING_FILTER "scaling-filter"
#define NVDSPREPROCESS_PROPERTY_SCALING_POOL_COMPUTE_HW "scaling-pool-compute-hw"
#define NVDSPREPROCESS_PROPERTY_SCALING_POOL_MEMORY_TYPE "scaling-pool-memory-type"

#define NVDSPREPROCESS_PROPERTY_NETWORK_INPUT_ORDER "network-input-order"
#define NVDSPREPROCESS_PROPERTY_NETWORK_SHAPE "network-input-shape"
#define NVDSPREPROCESS_PROPERTY_NETWORK_COLOR_FORMAT "network-color-format"
#define NVDSPREPROCESS_PROPERTY_TENSOR_DATA_TYPE "tensor-data-type"
#define NVDSPREPROCESS_PROPERTY_TENSOR_NAME "tensor-name"

#define NVDSPREPROCESS_PROPERTY_CUSTOM_LIB_NAME "custom-lib-path"
#define NVDSPREPROCESS_PROPERTY_TENSOR_PREPARATION_FUNCTION "custom-tensor-preparation-function"

#define NVDSPREPROCESS_USER_CONFIGS "user-configs"

#define NVDSPREPROCESS_GROUP "group-"
#define NVDSPREPROCESS_GROUP_SRC_IDS "src-ids"
#define NVDSPREPROCESS_GROUP_PROCESS_ON_ROI "process-on-roi"
#define NVDSPREPROCESS_GROUP_ROI_PARAMS_SRC "roi-params-src"
#define NVDSPREPROCESS_GROUP_CUSTOM_INPUT_PREPROCESS_FUNCTION "custom-input-transformation-function"

/**
 * Get GstNvDsPreProcessMemory structure associated with buffer allocated using
 * GstNvDsPreProcessAllocator.
 *
 * @param nvdspreprocess pointer to GstNvDsPreProcess structure
 *
 * @param cfg_file_path config file path
 *
 * @return boolean denoting if successfully parsed config file
 */
gboolean nvdspreprocess_parse_config_file(GstNvDsPreProcess *nvdspreprocess, gchar *cfg_file_path);

#endif /* NVDSPREPROCESS_PROPERTY_FILE_PARSER_H_ */
