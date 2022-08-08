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

#ifndef __GSTNVDSPREPROCESSALLOCATOR_H__
#define __GSTNVDSPREPROCESSALLOCATOR_H__

#include <cuda_runtime_api.h>
#include <gst/gst.h>

#include <vector>

#include "cudaEGL.h"
#include "nvbufsurface.h"

/**
 * This file describes the custom memory allocator for the Gstreamer TensorRT
 * plugin. The allocator allocates memory for a specified batch_size of frames
 * of resolution equal to the network input resolution and RGBA color format.
 * The frames are allocated on device memory.
 */

/**
 * Holds the pointer for the allocated memory.
 */
typedef struct {
    /** surface corresponding to memory allocated */
    NvBufSurface *surf;
    /** Vector of cuda resources created by registering the above egl images in CUDA. */
    std::vector<CUgraphicsResource> cuda_resources;
    /** Vector of CUDA eglFrames created by mapping the above cuda resources. */
    std::vector<CUeglFrame> egl_frames;
    /** Pointer to the memory allocated for the batch of frames (DGPU). */
    void *dev_memory_ptr;
    /** Vector of pointer to individual frame memories in the batch memory */
    std::vector<void *> frame_memory_ptrs;
} GstNvDsPreProcessMemory;

/**
 * Get GstNvDsPreProcessMemory structure associated with buffer allocated using
 * GstNvDsPreProcessAllocator.
 *
 * @param buffer GstBuffer allocated by this allocator.
 *
 * @return Pointer to the associated GstNvDsPreProcessMemory structure
 */
GstNvDsPreProcessMemory *gst_nvdspreprocess_buffer_get_memory(GstBuffer *buffer);

/**
 * structure containing video buffer allocator info
 */
typedef struct {
    /** video buffer width */
    guint width;
    /** video buffer height */
    guint height;
    /** color format */
    NvBufSurfaceColorFormat color_format;
    /** batch size */
    guint batch_size;
    /** memory type of buffer */
    NvBufSurfaceMemType memory_type;
} GstNvDsPreProcessVideoBufferAllocatorInfo;

/**
 * Create a new GstNvDsPreProcessAllocator with the given parameters.
 *
 * @param info video buffer allocator info.
 * @param raw_buf_size size of raw buffer to allocate.
 * @param gpu_id ID of the gpu where the batch memory will be allocated.
 * @param debug_tensor boolean to denote if DEBUG_TENSOR flag is enabled.
 *
 * @return Pointer to the GstNvDsPreProcessAllocator structure cast as GstAllocator
 */
GstAllocator *gst_nvdspreprocess_allocator_new(GstNvDsPreProcessVideoBufferAllocatorInfo *info,
                                               size_t raw_buf_size,
                                               guint gpu_id,
                                               gboolean debug_tensor);

#endif
