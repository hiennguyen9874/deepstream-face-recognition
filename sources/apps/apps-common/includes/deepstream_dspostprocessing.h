/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef _NVGSTDS_DSPOSTPROCESSING_H_
#define _NVGSTDS_DSPOSTPROCESSING_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <gst/gst.h>

typedef struct {
    // Create a bin for the element only if enabled
    gboolean enable;
    // Struct members to store config / properties for the element
    guint unique_id;
    guint gpu_id;
    gint *list_initial_colors;
    gsize num_initial_colors;
    // For nvvidconv
    guint nvbuf_memory_type;
} NvDsDsPostProcessingConfig;

// Struct to store references to the bin and elements
typedef struct {
    GstElement *bin;
    GstElement *queue;
    GstElement *pre_conv;
    GstElement *cap_filter;
    GstElement *elem_dspostprocessing;
} NvDsDsPostProcessingBin;

// Function to create the bin and set properties
gboolean create_dspostprocessing_bin(NvDsDsPostProcessingConfig *config,
                                     NvDsDsPostProcessingBin *bin);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_DSPOSTPROCESSING_H_ */
