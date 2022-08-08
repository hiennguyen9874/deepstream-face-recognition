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

#ifndef __GST_NVDSAUDIOTEMPLATE_META_H__
#define __GST_NVDSAUDIOTEMPLATE_META_H__

#include <gst/gst.h>

#include <iostream>

GType gst_mpeg_video_meta_api_get_type(void);
#define GST_AUDIO_TEMPLATE_META_API_TYPE (gst_audio_template_meta_api_get_type())
#define GST_AUDIO_TEMPLATE_META_INFO (gst_audio_template_meta_get_info())

typedef struct _GstAudioTemplateMeta {
    GstMeta meta;
    guint frame_count;
    guint custom_metadata_type;
    guint custom_metadata_size;
    void *custom_metadata_ptr;
} GstAudioTemplateMeta;

GType gst_audio_template_meta_api_get_type(void);
const GstMetaInfo *gst_audio_template_meta_get_info(void);

#endif /*__GST_NVDSAUDIOTEMPLATE_META_H__*/
