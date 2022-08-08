/*
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

/**
 * @file nvdspreprocess_meta.h
 * <b>NVIDIA DeepStream GStreamer NvDsPreProcess meta Specification </b>
 *
 * @b Description: This file specifies the metadata attached by
 * the DeepStream GStreamer NvDsPreProcess Plugin.
 */

/**
 * @defgroup   gstreamer_nvdspreprocess_api  NvDsPreProcess Plugin
 * Defines an API for the GStreamer NvDsPreProcess plugin.
 * @ingroup custom_gstreamer
 * @{
 */

#ifndef __NVDSPREPROCESS_META_H__
#define __NVDSPREPROCESS_META_H__

#include <string>
#include <vector>

#include "nvbufsurface.h"
#include "nvds_roi_meta.h"

/**
 * tensor meta containing prepared tensor and related info
 * inside preprocess user meta which is attached at batch level
 */
typedef struct {
    /** raw tensor buffer preprocessed for infer */
    void *raw_tensor_buffer;

    /** size of raw tensor buffer */
    guint64 buffer_size;

    /** raw tensor buffer shape */
    std::vector<int> tensor_shape;

    /** model datatype for which tensor prepared */
    NvDsDataType data_type;

    /** to be same as model input layer name */
    std::string tensor_name;

    /** gpu-id on which tensor prepared */
    guint gpu_id;

    /** pointer to buffer from tensor pool */
    void *private_data;

} NvDsPreProcessTensorMeta;

/**
 * preprocess meta as a user meta which is attached at
 * batch level
 */
typedef struct {
    /** target unique ids for which meta is prepared */
    std::vector<guint64> target_unique_ids;

    /** pointer to tensor meta */
    NvDsPreProcessTensorMeta *tensor_meta;

    /** list of roi vectors per batch */
    std::vector<NvDsRoiMeta> roi_vector;

    /** pointer to buffer from scaling pool*/
    void *private_data;

} GstNvDsPreProcessBatchMeta;

#endif /* __NVDSPREPROCESS_META_H__ */
