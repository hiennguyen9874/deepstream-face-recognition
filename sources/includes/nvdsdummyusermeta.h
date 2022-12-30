/**
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef __GST_NVDSDUMMYUSER_META_H__
#define __GST_NVDSDUMMYUSER_META_H__

#include <nvdsmeta.h>

#define NVDS_DUMMY_BBOX_META (nvds_get_user_meta_type((gchar *)"NVIDIA.DUMMY.BBOX.META"))

typedef enum _payload_type {
    NVDS_PAYLOAD_TYPE_DUMMY_BBOX = NVDS_START_USER_META + 4096,
} payload_type;

typedef struct faceboxes {
    float x;
    float y;
    float width;
    float height;
} faceboxes;

#endif //__GST_NVDSDUMMYUSER_META_H__
