/**
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef __GST_NVDSAUDIOTEMPLATE_H__
#define __GST_NVDSAUDIOTEMPLATE_H__

#include <cuda_runtime.h>
#include <glib-object.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>

#include <vector>

#include "gstnvdsmeta.h"
#include "nvdscustomlib_factory.hpp"
#include "nvdscustomlib_interface.hpp"
#include "nvtx3/nvToolsExt.h"

/* Package and library details required for plugin_init */
#define PACKAGE "nvdsaudiotemplate"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA example Template Plugin for integration with DeepStream on DGPU/Jetson"
#define BINARY_PACKAGE "NVIDIA DeepStream Template Plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS
/* Standard boilerplate stuff */
typedef struct _GstNvDsAudioTemplate GstNvDsAudioTemplate;
typedef struct _GstNvDsAudioTemplateClass GstNvDsAudioTemplateClass;

/* Standard boilerplate stuff */
#define GST_TYPE_NVDSAUDIOTEMPLATE (gst_nvdsaudiotemplate_get_type())
#define GST_NVDSAUDIOTEMPLATE(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDSAUDIOTEMPLATE, GstNvDsAudioTemplate))
#define GST_NVDSAUDIOTEMPLATE_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVDSAUDIOTEMPLATE, GstNvDsAudioTemplateClass))
#define GST_NVDSAUDIOTEMPLATE_GET_CLASS(obj) \
    (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_NVDSAUDIOTEMPLATE, GstNvDsAudioTemplateClass))
#define GST_IS_NVDSAUDIOTEMPLATE(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVDSAUDIOTEMPLATE))
#define GST_IS_NVDSAUDIOTEMPLATE_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVDSAUDIOTEMPLATE))
#define GST_NVDSAUDIOTEMPLATE_CAST(obj) ((GstNvDsAudioTemplate *)(obj))

struct _GstNvDsAudioTemplate {
    GstBaseTransform base_trans;

    /** Custom Library Factory and Interface */
    DSCustomLibrary_Factory *algo_factory;
    IDSCustomLibrary *algo_ctx;

    /** Custom Library Name and output caps string */
    gchar *custom_lib_name;

    /* Store custom lib property values */
    std::vector<Property> *vecProp;
    gchar *custom_prop_string;

    /** Boolean to signal output thread to stop. */
    gboolean stop;

    /** Input and Output audio info (resolution, color format, framerate, etc) */
    GstAudioInfo in_audio_info;
    GstAudioInfo out_audio_info;

    /** GPU ID on which we expect to execute the task */
    guint gpu_id;

    /** NVTX Domain. */
    nvtxDomainHandle_t nvtx_domain;

    GstCaps *sinkcaps;
    GstCaps *srccaps;

    guint frame_number;
    guint num_batch_buffers;
};

/** Boiler plate stuff */
struct _GstNvDsAudioTemplateClass {
    GstBaseTransformClass parent_class;
};

GType gst_nvdsaudiotemplate_get_type(void);

G_END_DECLS
#endif /* __GST_NVDSAUDIOTEMPLATE_H__ */
