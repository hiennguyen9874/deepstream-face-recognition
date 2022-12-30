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
 * @file
 * <b>NVIDIA GStreamer DeepStream: ROI Meta used in nvdspreprocess plugin </b>
 *
 * @b Description: This file defines the Metadata structure used to
 * carry DeepStream ROI metadata in GStreamer pipeline.
 *
 * @defgroup  ee_nvdspreprocess_api_group Pre-Process Metadata
 *
 * Specifies metadata concerning ROIs used in nvdspreprocess plugin.
 *
 * @ingroup NvDsPreProcessApi
 * @{
 */

#ifndef __NVDS_ROI_META_H__
#define __NVDS_ROI_META_H__

#include "nvdsmeta.h"
/////////////////
/* Start Custom */
/////////////////
#include "nvll_osd_struct.h"
////////////////
/* End Custom */
////////////////

/** max polygon points ; currently not being used */
#define DS_MAX_POLYGON_POINTS 8

/** DS NvBufSurfaceParams */
typedef struct NvBufSurfaceParams NvBufSurfaceParams;

/** DS NvDsFrameMeta */
typedef struct _NvDsFrameMeta NvDsFrameMeta;

/** classifier meta list */
typedef GList NvDsClassifierMetaList;

/** user meta list */
typedef GList NvDsUserMetaList;

/**
 * Data type used for model in infer
 */
typedef enum {
    /** FP32 data type */
    NvDsDataType_FP32,
    /** UINT8 data type */
    NvDsDataType_UINT8,
    /** INT8 data type */
    NvDsDataType_INT8,
    /** UINT32 data type */
    NvDsDataType_UINT32,
    /** INT32 data type */
    NvDsDataType_INT32,
    /** FP16 data type */
    NvDsDataType_FP16,
} NvDsDataType;

/**
 * Unit Type Fullframe/ROI/Crop Objects
 */
typedef enum {
    /** Full frames */
    NvDsUnitType_FullFrame = 0,
    /** Region of Interests (ROIs) */
    NvDsUnitType_ROI,
    /** object mode */
    NvDsUnitType_Object,
} NvDsUnitType;

/**
 * Holds Information about ROI Metadata
 */
typedef struct NvDsRoiMeta {
    /* per roi information */
    NvOSD_RectParams roi;

    /** currently not being used */
    guint roi_polygon[DS_MAX_POLYGON_POINTS][2];

    /* Scaled & converted buffer to processing width/height */
    NvBufSurfaceParams *converted_buffer;

    /* Deepstream frame meta */
    NvDsFrameMeta *frame_meta;

    /** Ratio by which the frame/ROI crop was scaled in horizontal direction
     * Required when scaling co-ordinates/sizes in metadata
     * back to input resolution. */
    gdouble scale_ratio_x;

    /** Ratio by which the frame/ROI crop was scaled in vertical direction
     * Required when scaling co-ordinates/sizes in metadata
     * back to input resolution. */
    gdouble scale_ratio_y;

    /** offsets in horizontal direction while scaling */
    gdouble offset_left;

    /** offsets in vertical direction while scaling */
    gdouble offset_top;

    /** Holds a pointer to a list of pointers of type @ref NvDsClassifierMeta. */
    NvDsClassifierMetaList *classifier_meta_list;

    /** Holds a pointer to a list of pointers of type @ref NvDsUserMeta. */
    NvDsUserMetaList *roi_user_meta_list;

} NvDsRoiMeta;

#endif
