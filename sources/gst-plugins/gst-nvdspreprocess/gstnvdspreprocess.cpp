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

#include "gstnvdspreprocess.h"

#include <string.h>
#include <sys/time.h>

#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>

#include "gstnvdspreprocess_allocator.h"
#include "nvdspreprocess_property_parser.h"

GST_DEBUG_CATEGORY_STATIC(gst_nvdspreprocess_debug);
#define GST_CAT_DEFAULT gst_nvdspreprocess_debug
#define USE_EGLIMAGE 1

/** compile makefile WITH_OPENCV:=1
 * and enable this to write transformed ROIs to files
 */
//#define DUMP_ROIS

#ifdef DUMP_ROIS
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#endif

/**
 * enable to debug tensor prepared by this plugin
 * and dump it in .bin files
 */
//#define DEBUG_TENSOR
/**
 * enable to draw ROIs on display
 */
#define DRAW_ROIS

/* Enum to identify properties */
enum {
    PROP_0,
    PROP_UNIQUE_ID,
    PROP_ENABLE,
    PROP_PROCESSING_WIDTH,
    PROP_PROCESSING_HEIGHT,
    PROP_GPU_DEVICE_ID,
    PROP_CONFIG_FILE
};

#define CHECK_NVDS_MEMORY_AND_GPUID(object, surface)                                             \
    ({                                                                                           \
        int _errtype = 0;                                                                        \
        do {                                                                                     \
            if ((surface->memType == NVBUF_MEM_DEFAULT ||                                        \
                 surface->memType == NVBUF_MEM_CUDA_DEVICE) &&                                   \
                (surface->gpuId != object->gpu_id)) {                                            \
                GST_ELEMENT_ERROR(                                                               \
                    object, RESOURCE, FAILED,                                                    \
                    ("Input surface gpu-id doesnt match with configured gpu-id for element,"     \
                     " please allocate input using unified memory, or use same gpu-ids"),        \
                    ("surface-gpu-id=%d,%s-gpu-id=%d", surface->gpuId, GST_ELEMENT_NAME(object), \
                     object->gpu_id));                                                           \
                _errtype = 1;                                                                    \
            }                                                                                    \
        } while (0);                                                                             \
        _errtype;                                                                                \
    })

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_PROCESSING_WIDTH 640
#define DEFAULT_PROCESSING_HEIGHT 480
#define DEFAULT_GPU_ID 0
#define DEFAULT_BATCH_SIZE 1
#define DEFAULT_CONFIG_FILE_PATH ""
#define DEFAULT_SCALING_POOL_COMPUTE_HW NvBufSurfTransformCompute_Default
#define DEFAULT_SCALING_BUF_POOL_SIZE 6 /** Inter Buffer Pool Size for Scale & Converted ROIs */
#define DEFAULT_TENSOR_BUF_POOL_SIZE 6  /** Tensor Buffer Pool Size */

#define RGB_BYTES_PER_PIXEL 3
#define RGBA_BYTES_PER_PIXEL 4
#define Y_BYTES_PER_PIXEL 1
#define UV_BYTES_PER_PIXEL 2

#define MIN_INPUT_OBJECT_WIDTH 16
#define MIN_INPUT_OBJECT_HEIGHT 16

#define CHECK_NPP_STATUS(npp_status, error_str)                                                  \
    do {                                                                                         \
        if ((npp_status) != NPP_SUCCESS) {                                                       \
            g_print("Error: %s in %s at line %d: NPP Error %d\n", error_str, __FILE__, __LINE__, \
                    npp_status);                                                                 \
            goto error;                                                                          \
        }                                                                                        \
    } while (0)

#define CHECK_CUDA_STATUS(cuda_status, error_str)                                       \
    do {                                                                                \
        if ((cuda_status) != cudaSuccess) {                                             \
            g_print("Error: %s in %s at line %d (%s)\n", error_str, __FILE__, __LINE__, \
                    cudaGetErrorName(cuda_status));                                     \
            goto error;                                                                 \
        }                                                                               \
    } while (0)

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_nvdspreprocess_sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_nvdspreprocess_src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvdspreprocess_parent_class parent_class
G_DEFINE_TYPE(GstNvDsPreProcess, gst_nvdspreprocess, GST_TYPE_BASE_TRANSFORM);

static void gst_nvdspreprocess_set_property(GObject *object,
                                            guint prop_id,
                                            const GValue *value,
                                            GParamSpec *pspec);
static void gst_nvdspreprocess_get_property(GObject *object,
                                            guint prop_id,
                                            GValue *value,
                                            GParamSpec *pspec);

static gboolean gst_nvdspreprocess_set_caps(GstBaseTransform *btrans,
                                            GstCaps *incaps,
                                            GstCaps *outcaps);
static gboolean gst_nvdspreprocess_start(GstBaseTransform *btrans);
static gboolean gst_nvdspreprocess_stop(GstBaseTransform *btrans);

static GstFlowReturn gst_nvdspreprocess_submit_input_buffer(GstBaseTransform *btrans,
                                                            gboolean discont,
                                                            GstBuffer *inbuf);
static GstFlowReturn gst_nvdspreprocess_generate_output(GstBaseTransform *btrans,
                                                        GstBuffer **outbuf);
static gpointer gst_nvdspreprocess_output_loop(gpointer data);

template <class T>
T *dlsym_ptr(void *handle, char const *name)
{
    return reinterpret_cast<T *>(dlsym(handle, name));
}

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void gst_nvdspreprocess_class_init(GstNvDsPreProcessClass *klass)
{
    GObjectClass *gobject_class;
    GstElementClass *gstelement_class;
    GstBaseTransformClass *gstbasetransform_class;

    // Indicates we want to use DS buf api
    g_setenv("DS_NEW_BUFAPI", "1", TRUE);

    gobject_class = (GObjectClass *)klass;
    gstelement_class = (GstElementClass *)klass;
    gstbasetransform_class = (GstBaseTransformClass *)klass;

    /* Overide base class functions */
    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_nvdspreprocess_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_nvdspreprocess_get_property);

    gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR(gst_nvdspreprocess_set_caps);
    gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_nvdspreprocess_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_nvdspreprocess_stop);

    gstbasetransform_class->submit_input_buffer =
        GST_DEBUG_FUNCPTR(gst_nvdspreprocess_submit_input_buffer);
    gstbasetransform_class->generate_output = GST_DEBUG_FUNCPTR(gst_nvdspreprocess_generate_output);

    /* Install properties */
    g_object_class_install_property(
        gobject_class, PROP_UNIQUE_ID,
        g_param_spec_uint("unique-id", "Unique ID",
                          "Unique ID for the element. Can be used to identify output of the"
                          " element",
                          0, G_MAXUINT, DEFAULT_UNIQUE_ID,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_ENABLE,
        g_param_spec_boolean(
            "enable", "Enable", "Enable gst-nvdspreprocess plugin, or set in passthrough mode",
            TRUE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, PROP_GPU_DEVICE_ID,
        g_param_spec_uint(
            "gpu-id", "Set GPU Device ID", "Set GPU Device ID", 0, G_MAXUINT, 0,
            GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, PROP_CONFIG_FILE,
        g_param_spec_string("config-file", "Preprocess Config File", "Preprocess Config File",
                            DEFAULT_CONFIG_FILE_PATH,
                            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    /* Set sink and src pad capabilities */
    gst_element_class_add_pad_template(
        gstelement_class, gst_static_pad_template_get(&gst_nvdspreprocess_src_template));
    gst_element_class_add_pad_template(
        gstelement_class, gst_static_pad_template_get(&gst_nvdspreprocess_sink_template));

    /* Set metadata describing the element */
    gst_element_class_set_details_simple(
        gstelement_class, "gst-nvdspreprocess plugin", "gst-nvdspreprocess plugin",
        "Preprocessing using custom algorithms for different streams",
        "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
        "@ https://devtalk.nvidia.com/default/board/209/");
}

static void gst_nvdspreprocess_init(GstNvDsPreProcess *nvdspreprocess)
{
    GstBaseTransform *btrans = GST_BASE_TRANSFORM(nvdspreprocess);

    /* We will not be generating a new buffer. Just adding / updating
     * metadata. */
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(btrans), TRUE);
    /* We do not want to change the input caps. Set to passthrough. transform_ip
     * is still called. */
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(btrans), TRUE);

    /* Initialize all property variables to default values */
    nvdspreprocess->unique_id = DEFAULT_UNIQUE_ID;
    nvdspreprocess->enable = TRUE;
    nvdspreprocess->processing_width = DEFAULT_PROCESSING_WIDTH;
    nvdspreprocess->processing_height = DEFAULT_PROCESSING_HEIGHT;
    nvdspreprocess->gpu_id = DEFAULT_GPU_ID;
    nvdspreprocess->max_batch_size = DEFAULT_BATCH_SIZE;
    nvdspreprocess->scaling_pool_compute_hw = DEFAULT_SCALING_POOL_COMPUTE_HW;
    nvdspreprocess->config_file_path = g_strdup(DEFAULT_CONFIG_FILE_PATH);
    nvdspreprocess->config_file_parse_successful = FALSE;
    nvdspreprocess->scaling_buf_pool_size = DEFAULT_SCALING_BUF_POOL_SIZE;
    nvdspreprocess->tensor_buf_pool_size = DEFAULT_TENSOR_BUF_POOL_SIZE;

    /* Set the default pre-processing transform params. */
    nvdspreprocess->transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
    nvdspreprocess->transform_params.transform_filter = NvBufSurfTransformInter_Default;
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void gst_nvdspreprocess_set_property(GObject *object,
                                            guint prop_id,
                                            const GValue *value,
                                            GParamSpec *pspec)
{
    GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS(object);
    switch (prop_id) {
    case PROP_UNIQUE_ID:
        nvdspreprocess->unique_id = g_value_get_uint(value);
        break;
    case PROP_ENABLE:
        nvdspreprocess->enable = g_value_get_boolean(value);
        break;
    case PROP_GPU_DEVICE_ID:
        nvdspreprocess->gpu_id = g_value_get_uint(value);
        break;
    case PROP_CONFIG_FILE: {
        g_mutex_lock(&nvdspreprocess->preprocess_lock);
        g_free(nvdspreprocess->config_file_path);
        nvdspreprocess->config_file_path = g_value_dup_string(value);
        /* Parse the initialization parameters from the config file. This function
         * gives preference to values set through the set_property function over
         * the values set in the config file. */
        nvdspreprocess->config_file_parse_successful =
            nvdspreprocess_parse_config_file(nvdspreprocess, nvdspreprocess->config_file_path);
        if (nvdspreprocess->config_file_parse_successful) {
            GST_DEBUG_OBJECT(nvdspreprocess, "Successfully Parsed Config file\n");
        }
        g_mutex_unlock(&nvdspreprocess->preprocess_lock);
    } break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void gst_nvdspreprocess_get_property(GObject *object,
                                            guint prop_id,
                                            GValue *value,
                                            GParamSpec *pspec)
{
    GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS(object);

    switch (prop_id) {
    case PROP_UNIQUE_ID:
        g_value_set_uint(value, nvdspreprocess->unique_id);
        break;
    case PROP_ENABLE:
        g_value_set_boolean(value, nvdspreprocess->enable);
        break;
    case PROP_GPU_DEVICE_ID:
        g_value_set_uint(value, nvdspreprocess->gpu_id);
        break;
    case PROP_CONFIG_FILE:
        g_value_set_string(value, nvdspreprocess->config_file_path);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/**
 * Initialize all resources and start the process thread
 */
static gboolean gst_nvdspreprocess_start(GstBaseTransform *btrans)
{
    GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS(btrans);
    std::string nvtx_str;
    NvBufSurfaceColorFormat color_format;
    cudaError_t cudaReturn;
    /** scaling pool params */
    GstStructure *scaling_pool_config;
    GstAllocator *scaling_pool_allocator;
    GstAllocationParams allocation_params;
    /** tensor pool params */
    GstStructure *tensor_pool_config;
    GstAllocator *tensor_pool_allocator;
    GstAllocationParams tensor_pool_allocation_params;

    if (!nvdspreprocess->config_file_path || strlen(nvdspreprocess->config_file_path) == 0) {
        GST_ELEMENT_ERROR(nvdspreprocess, LIBRARY, SETTINGS, ("Configuration file not provided"),
                          (nullptr));
        return FALSE;
    }

    if (nvdspreprocess->config_file_parse_successful == FALSE) {
        GST_ELEMENT_ERROR(nvdspreprocess, LIBRARY, SETTINGS, ("Configuration file parsing failed"),
                          ("Config file path: %s", nvdspreprocess->config_file_path));
        return FALSE;
    }

    nvtx_str = "GstNvDsPreProcess: UID=" + std::to_string(nvdspreprocess->unique_id);
    auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy(d); };
    std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr(
        nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);

    nvdspreprocess->custom_initparams.tensor_params = nvdspreprocess->tensor_params;
    nvdspreprocess->custom_initparams.unique_id = nvdspreprocess->unique_id;
    nvdspreprocess->custom_initparams.config_file_path = nvdspreprocess->config_file_path;

    /* Initialize custom library */
    if (nvdspreprocess->custom_lib_path) {
        nvdspreprocess->custom_lib_handle = dlopen(nvdspreprocess->custom_lib_path, RTLD_NOW);
        std::function<CustomCtx *(CustomInitParams)> initLib;

        if (nvdspreprocess->custom_lib_handle) {
            GST_DEBUG_OBJECT(nvdspreprocess, "Custom Library Opened Successfully\n");
            initLib = dlsym_ptr<CustomCtx *(CustomInitParams)>(nvdspreprocess->custom_lib_handle,
                                                               "initLib");
            nvdspreprocess->custom_lib_ctx = initLib(nvdspreprocess->custom_initparams);
            if (nvdspreprocess->custom_lib_ctx == nullptr) {
                GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                                  ("Error while initializing Custom Library\n"), (NULL));
                return FALSE;
            }

            if (nvdspreprocess->custom_tensor_function_name) {
                nvdspreprocess->custom_tensor_function = dlsym_ptr<NvDsPreProcessStatus(
                    CustomCtx *, NvDsPreProcessBatch *, NvDsPreProcessCustomBuf *&,
                    CustomTensorParams &, NvDsPreProcessAcquirer *)>(
                    nvdspreprocess->custom_lib_handle, nvdspreprocess->custom_tensor_function_name);

                if (!nvdspreprocess->custom_tensor_function) {
                    GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                                      ("Error while loading Custom Tensor Preparation function\n"),
                                      (NULL));
                    return FALSE;
                }
            }

            for (guint gcnt = 0; gcnt < nvdspreprocess->nvdspreprocess_groups.size(); gcnt++) {
                if (nvdspreprocess->nvdspreprocess_groups[gcnt]->custom_transform_function_name) {
                    nvdspreprocess->nvdspreprocess_groups[gcnt]->custom_transform =
                        dlsym_ptr<NvDsPreProcessStatus(NvBufSurface *, NvBufSurface *,
                                                       CustomTransformParams &)>(
                            nvdspreprocess->custom_lib_handle,
                            nvdspreprocess->nvdspreprocess_groups[gcnt]
                                ->custom_transform_function_name);

                    if (!nvdspreprocess->nvdspreprocess_groups[gcnt]->custom_transform) {
                        GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                                          ("Custom Transformation function not found"), (NULL));
                        return FALSE;
                    }
                }
            }
        } else {
            GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED, ("Could not open custom library\n"),
                              (NULL));
            return FALSE;
        }
    }

    GST_DEBUG_OBJECT(nvdspreprocess, "Initialized Custom Library Context\n");

    /* Create the intermediate NvBufSurface structure for holding an array of input
     * NvBufSurfaceParams for batched transforms. */
    nvdspreprocess->batch_insurf.surfaceList =
        new NvBufSurfaceParams[nvdspreprocess->max_batch_size];
    nvdspreprocess->batch_insurf.batchSize = nvdspreprocess->max_batch_size;
    nvdspreprocess->batch_insurf.gpuId = nvdspreprocess->gpu_id;

    /* Holds output of batched transforms. */
    nvdspreprocess->batch_outsurf.surfaceList =
        new NvBufSurfaceParams[nvdspreprocess->max_batch_size];
    nvdspreprocess->batch_outsurf.batchSize = nvdspreprocess->max_batch_size;
    nvdspreprocess->batch_outsurf.gpuId = nvdspreprocess->gpu_id;

    /* Set the NvBufSurfTransform config parameters. */
    nvdspreprocess->transform_config_params.gpu_id = nvdspreprocess->gpu_id;
    nvdspreprocess->transform_config_params.cuda_stream = nvdspreprocess->convert_stream;
    nvdspreprocess->transform_config_params.compute_mode = nvdspreprocess->scaling_pool_compute_hw;

    /* Set up the NvBufSurfTransformParams structure for batched transforms. */
    nvdspreprocess->transform_params.src_rect =
        new NvBufSurfTransformRect[nvdspreprocess->max_batch_size];
    nvdspreprocess->transform_params.dst_rect =
        new NvBufSurfTransformRect[nvdspreprocess->max_batch_size];
    nvdspreprocess->transform_params.transform_flag =
        NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC | NVBUFSURF_TRANSFORM_CROP_DST;
    nvdspreprocess->transform_params.transform_flip = NvBufSurfTransform_None;
    nvdspreprocess->transform_params.transform_filter =
        nvdspreprocess->scaling_pool_interpolation_filter;

    /* Create a buffer pool for internal memory required for scaling frames to
     * network resolution / cropping objects. The pool allocates
     * nvdspreprocess->scaling_buf_pool_size buffers at start and keeps reusing them. */
    nvdspreprocess->scaling_pool = gst_buffer_pool_new();
    scaling_pool_config = gst_buffer_pool_get_config(nvdspreprocess->scaling_pool);
    gst_buffer_pool_config_set_params(scaling_pool_config, nullptr, sizeof(GstNvDsPreProcessMemory),
                                      nvdspreprocess->scaling_buf_pool_size,
                                      nvdspreprocess->scaling_buf_pool_size);

    /* Based on the network input requirements decide the buffer pool color format. */
    switch (nvdspreprocess->tensor_params.network_color_format) {
    case NvDsPreProcessFormat_RGB:
    case NvDsPreProcessFormat_BGR:
        if (nvdspreprocess->transform_config_params.compute_mode == NvBufSurfTransformCompute_VIC
#ifdef __aarch64__
            || nvdspreprocess->transform_config_params.compute_mode ==
                   NvBufSurfTransformCompute_Default
#endif
        ) {
            g_print("Warning: converting ROIs to RGBA for VIC mode\n");
            color_format = NVBUF_COLOR_FORMAT_RGBA;
            nvdspreprocess->scaling_pool_format = NvDsPreProcessFormat_RGBA;
        } else {
            color_format = NVBUF_COLOR_FORMAT_RGB;
            nvdspreprocess->scaling_pool_format = NvDsPreProcessFormat_RGB;
        }
        break;
    case NvDsPreProcessFormat_GRAY:
        if (nvdspreprocess->transform_config_params.compute_mode == NvBufSurfTransformCompute_VIC
#ifdef __aarch64__
            || nvdspreprocess->transform_config_params.compute_mode ==
                   NvBufSurfTransformCompute_Default
#endif
        ) {
            g_print("Warning: converting ROIs to NV12 for VIC mode\n");
            color_format = NVBUF_COLOR_FORMAT_NV12;
            nvdspreprocess->scaling_pool_format = NvDsPreProcessFormat_GRAY;
        } else {
            color_format = NVBUF_COLOR_FORMAT_GRAY8;
            nvdspreprocess->scaling_pool_format = NvDsPreProcessFormat_GRAY;
        }
        break;
    default:
        GST_ELEMENT_ERROR(nvdspreprocess, LIBRARY, SETTINGS,
                          ("Unsupported network color format: %d",
                           nvdspreprocess->tensor_params.network_color_format),
                          (nullptr));
        goto error;
    }

    GST_DEBUG_OBJECT(nvdspreprocess, "network-input-order = %d\n",
                     nvdspreprocess->custom_initparams.tensor_params.network_input_order);

    if (nvdspreprocess->tensor_params.network_input_order ==
        NvDsPreProcessNetworkInputOrder_kNCHW) {
        if (nvdspreprocess->processing_height !=
                nvdspreprocess->tensor_params.network_input_shape[2] ||
            nvdspreprocess->processing_width !=
                nvdspreprocess->tensor_params.network_input_shape[3]) {
            GST_ELEMENT_ERROR(nvdspreprocess, LIBRARY, SETTINGS,
                              ("Processing height = %d and width =%d should be same as network "
                               "height = %d and width =%d\n",
                               nvdspreprocess->processing_height, nvdspreprocess->processing_width,
                               nvdspreprocess->tensor_params.network_input_shape[2],
                               nvdspreprocess->tensor_params.network_input_shape[3]),
                              (nullptr));
        }
    } else if (nvdspreprocess->tensor_params.network_input_order ==
               NvDsPreProcessNetworkInputOrder_kNHWC) {
        if (nvdspreprocess->processing_height !=
                nvdspreprocess->tensor_params.network_input_shape[1] ||
            nvdspreprocess->processing_width !=
                nvdspreprocess->tensor_params.network_input_shape[2]) {
            GST_ELEMENT_ERROR(nvdspreprocess, LIBRARY, SETTINGS,
                              ("Processing height = %d and width =%d should be same as network "
                               "height = %d and width =%d\n",
                               nvdspreprocess->processing_height, nvdspreprocess->processing_width,
                               nvdspreprocess->tensor_params.network_input_shape[1],
                               nvdspreprocess->tensor_params.network_input_shape[2]),
                              (nullptr));
        }
    } else if (nvdspreprocess->tensor_params.network_input_order ==
               NvDsPreProcessNetworkInputOrder_CUSTOM) {
        g_print("Using user provided processing height = %d and processing width = %d\n",
                nvdspreprocess->processing_height, nvdspreprocess->processing_width);
    } else {
        GST_ELEMENT_ERROR(nvdspreprocess, LIBRARY, SETTINGS,
                          ("Unsupported network input order: %d",
                           nvdspreprocess->tensor_params.network_input_order),
                          (nullptr));
    }

    /* Create a new GstNvDsPreProcess Allocator instance. Allocator has methods to allocate
     * and free custom memories. */
    GstNvDsPreProcessVideoBufferAllocatorInfo allocator_info;
    allocator_info.width = nvdspreprocess->processing_width;
    allocator_info.height = nvdspreprocess->processing_height;
    allocator_info.color_format = color_format;
    allocator_info.batch_size = nvdspreprocess->max_batch_size;

#ifdef DUMP_ROIS
    allocator_info.memory_type = NVBUF_MEM_CUDA_UNIFIED;
#else
    allocator_info.memory_type = nvdspreprocess->scaling_pool_memory_type;
#endif

    GST_DEBUG_OBJECT(nvdspreprocess, "Scaling pool batch-size = %d\n",
                     nvdspreprocess->max_batch_size);

    scaling_pool_allocator =
        gst_nvdspreprocess_allocator_new(&allocator_info, 1, nvdspreprocess->gpu_id, FALSE);

    memset(&allocation_params, 0, sizeof(allocation_params));

    gst_buffer_pool_config_set_allocator(scaling_pool_config, scaling_pool_allocator,
                                         &allocation_params);

    if (!gst_buffer_pool_set_config(nvdspreprocess->scaling_pool, scaling_pool_config)) {
        GST_ELEMENT_ERROR(nvdspreprocess, RESOURCE, FAILED,
                          ("Failed to set config on buffer scaling_pool"), (nullptr));
        goto error;
    }

    /* Start the buffer pool and allocate all internal buffers. */
    if (!gst_buffer_pool_set_active(nvdspreprocess->scaling_pool, TRUE)) {
        GST_ELEMENT_ERROR(nvdspreprocess, RESOURCE, FAILED, ("Failed to set buffer pool to active"),
                          (nullptr));
        goto error;
    }

    /* Create a buffer pool for internal memory required for scaling frames to
     * network resolution / cropping objects. The pool allocates
     * INTERNAL_BUF_POOL_SIZE buffers at start and keeps reusing them. */

    nvdspreprocess->tensor_pool = gst_buffer_pool_new();
    tensor_pool_config = gst_buffer_pool_get_config(nvdspreprocess->tensor_pool);
    gst_buffer_pool_config_set_params(tensor_pool_config, nullptr, sizeof(GstNvDsPreProcessMemory),
                                      nvdspreprocess->tensor_buf_pool_size,
                                      nvdspreprocess->tensor_buf_pool_size);

    nvdspreprocess->tensor_params.buffer_size = 1;
    for (auto &p : nvdspreprocess->tensor_params.network_input_shape) {
        nvdspreprocess->tensor_params.buffer_size *= p;
    }

    switch (nvdspreprocess->tensor_params.data_type) {
    case NvDsDataType_FP32:
    case NvDsDataType_UINT32:
    case NvDsDataType_INT32:
        nvdspreprocess->tensor_params.buffer_size *= 4;
        break;
    case NvDsDataType_UINT8:
    case NvDsDataType_INT8:
        nvdspreprocess->tensor_params.buffer_size *= 1;
        break;
    case NvDsDataType_FP16:
        nvdspreprocess->tensor_params.buffer_size *= 2;
        break;
    default:
        GST_ELEMENT_ERROR(nvdspreprocess, LIBRARY, SETTINGS,
                          ("Tensor data type : %d is not Supported\n",
                           (int)nvdspreprocess->tensor_params.data_type),
                          (nullptr));
        goto error;
    }

    GST_DEBUG_OBJECT(nvdspreprocess, "Tensor Buffer Pool size = %ld data-type=%d\n",
                     nvdspreprocess->tensor_params.buffer_size,
                     nvdspreprocess->tensor_params.data_type);
#ifdef DEBUG_TENSOR
    tensor_pool_allocator = gst_nvdspreprocess_allocator_new(
        NULL, nvdspreprocess->tensor_params.buffer_size, nvdspreprocess->gpu_id, TRUE);
#else
    tensor_pool_allocator = gst_nvdspreprocess_allocator_new(
        NULL, nvdspreprocess->tensor_params.buffer_size, nvdspreprocess->gpu_id, FALSE);
#endif
    memset(&tensor_pool_allocation_params, 0, sizeof(tensor_pool_allocation_params));

    gst_buffer_pool_config_set_allocator(tensor_pool_config, tensor_pool_allocator,
                                         &tensor_pool_allocation_params);

    if (!gst_buffer_pool_set_config(nvdspreprocess->tensor_pool, tensor_pool_config)) {
        GST_ELEMENT_ERROR(nvdspreprocess, RESOURCE, FAILED,
                          ("Failed to set config on tensor buffer pool"), (nullptr));
        goto error;
    }

    /* Start the buffer pool and allocate all internal buffers. */
    if (!gst_buffer_pool_set_active(nvdspreprocess->tensor_pool, TRUE)) {
        GST_ELEMENT_ERROR(nvdspreprocess, RESOURCE, FAILED,
                          ("Failed to set tensor buffer pool to active"), (nullptr));
        goto error;
    }

    cudaReturn = cudaSetDevice(nvdspreprocess->gpu_id);
    if (cudaReturn != cudaSuccess) {
        GST_ELEMENT_ERROR(nvdspreprocess, RESOURCE, FAILED,
                          ("Failed to set cuda device %d", nvdspreprocess->gpu_id),
                          ("cudaSetDevice failed with error %s", cudaGetErrorName(cudaReturn)));
        goto error;
    }

    cudaReturn = cudaStreamCreateWithFlags(&nvdspreprocess->convert_stream, cudaStreamNonBlocking);
    if (cudaReturn != cudaSuccess) {
        GST_ELEMENT_ERROR(
            nvdspreprocess, RESOURCE, FAILED,
            ("Failed to create cuda stream for conversion/transformation"),
            ("cudaStreamCreateWithFlags failed with error %s", cudaGetErrorName(cudaReturn)));
        goto error;
    }

    /** class for acquiring/releasing buffer from tensor pool */
    nvdspreprocess->acquire_impl =
        std::make_unique<NvDsPreProcessAcquirerImpl>(nvdspreprocess->tensor_pool);
    nvdspreprocess->tensor_buf = new NvDsPreProcessCustomBuf;

    nvdspreprocess->nvtx_domain = nvtx_domain_ptr.release();

    /* Create process queue to transfer data between threads.
     * We will be using this queue to maintain the list of frames/objects
     * currently given to the algorithm for processing. */
    nvdspreprocess->preprocess_queue = g_queue_new();

    /* Start a thread which will pop output from the algorithm, form NvDsMeta and
     * push buffers to the next element. */
    nvdspreprocess->output_thread = g_thread_new("nvdspreprocess-process-thread",
                                                 gst_nvdspreprocess_output_loop, nvdspreprocess);

    return TRUE;

error:
    delete[] nvdspreprocess->transform_params.src_rect;
    delete[] nvdspreprocess->transform_params.dst_rect;

    delete[] nvdspreprocess->batch_insurf.surfaceList;
    delete[] nvdspreprocess->batch_outsurf.surfaceList;

    if (nvdspreprocess->convert_stream) {
        cudaStreamDestroy(nvdspreprocess->convert_stream);
        nvdspreprocess->convert_stream = NULL;
    }

    return FALSE;
}

/**
 * Stop the process thread and free up all the resources
 */
static gboolean gst_nvdspreprocess_stop(GstBaseTransform *btrans)
{
    GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS(btrans);

    g_mutex_lock(&nvdspreprocess->preprocess_lock);

    /* Wait till all the items in the queue are handled. */
    while (!g_queue_is_empty(nvdspreprocess->preprocess_queue)) {
        g_cond_wait(&nvdspreprocess->preprocess_cond, &nvdspreprocess->preprocess_lock);
    }

    nvdspreprocess->stop = TRUE;

    g_cond_broadcast(&nvdspreprocess->preprocess_cond);
    g_mutex_unlock(&nvdspreprocess->preprocess_lock);

    g_thread_join(nvdspreprocess->output_thread);

    if (nvdspreprocess->convert_stream)
        cudaStreamDestroy(nvdspreprocess->convert_stream);
    nvdspreprocess->convert_stream = NULL;

    delete[] nvdspreprocess->transform_params.src_rect;
    delete[] nvdspreprocess->transform_params.dst_rect;

    delete[] nvdspreprocess->batch_insurf.surfaceList;
    delete[] nvdspreprocess->batch_outsurf.surfaceList;

    g_queue_free(nvdspreprocess->preprocess_queue);

    if (nvdspreprocess->config_file_path) {
        g_free(nvdspreprocess->config_file_path);
        nvdspreprocess->config_file_path = NULL;
    }

    /* Deinitialize custom library */
    if (nvdspreprocess->custom_lib_path) {
        std::function<void(void *)> deInitLib;

        if (nvdspreprocess->custom_lib_handle) {
            deInitLib = dlsym_ptr<void(void *)>(nvdspreprocess->custom_lib_handle, "deInitLib");
            deInitLib(nvdspreprocess->custom_lib_ctx);
        } else {
            GST_ELEMENT_ERROR(nvdspreprocess, RESOURCE, FAILED,
                              ("Could not open custom library for deinit\n"), (nullptr));
            return FALSE;
        }
    }
    dlclose(nvdspreprocess->custom_lib_handle);
    GST_DEBUG_OBJECT(nvdspreprocess, "Successfully Closed Custom Library\n");

    if (nvdspreprocess->custom_lib_path) {
        delete (nvdspreprocess->custom_lib_path);
        nvdspreprocess->custom_lib_path = NULL;
    }

    nvdspreprocess->acquire_impl.reset();

    if (nvdspreprocess->tensor_buf) {
        delete nvdspreprocess->tensor_buf;
        nvdspreprocess->tensor_buf = NULL;
    }

    /* delete the heap allocated memory */
    for (auto &group : nvdspreprocess->nvdspreprocess_groups) {
        group->framemeta_map.clear();
        delete group;
        group = NULL;
    }

    /* Free up the memory allocated by pool. */
    gst_object_unref(nvdspreprocess->scaling_pool);
    gst_object_unref(nvdspreprocess->tensor_pool);

    return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean gst_nvdspreprocess_set_caps(GstBaseTransform *btrans,
                                            GstCaps *incaps,
                                            GstCaps *outcaps)
{
    GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS(btrans);
    /* Save the input video information, since this will be required later. */
    // gst_video_info_from_caps (&nvdspreprocess->video_info, incaps);

    CHECK_CUDA_STATUS(cudaSetDevice(nvdspreprocess->gpu_id), "Unable to set cuda device");

    return TRUE;

error:
    return FALSE;
}

/**
 * Scale the entire frame to the processing resolution maintaining aspect ratio.
 * Or crop and scale objects to the processing resolution maintaining the aspect
 * ratio and fills data for batched conversation */
static GstFlowReturn scale_and_fill_data(GstNvDsPreProcess *nvdspreprocess,
                                         NvBufSurfaceParams *src_frame,
                                         NvOSD_RectParams *crop_rect_params,
                                         gdouble &ratio_x,
                                         gdouble &ratio_y,
                                         gdouble &offset_left,
                                         gdouble &offset_top,
                                         NvBufSurface *dest_surf,
                                         NvBufSurfaceParams *dest_frame,
                                         void *destCudaPtr)
{
    if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0)) {
        GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                          ("%s:crop_rect_params dimensions are zero", __func__), (NULL));
        return GST_FLOW_ERROR;
    }

    gint src_left = GST_ROUND_UP_2((unsigned int)crop_rect_params->left);
    gint src_top = GST_ROUND_UP_2((unsigned int)crop_rect_params->top);
    gint src_width = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->width);
    gint src_height = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->height);
    guint dest_width, dest_height;

    if (nvdspreprocess->maintain_aspect_ratio) {
        /* Calculate the destination width and height required to maintain
         * the aspect ratio. */
        double hdest = dest_frame->width * src_height / (double)src_width;
        double wdest = dest_frame->height * src_width / (double)src_height;
        int pixel_size;
        cudaError_t cudaReturn;

        if (hdest <= dest_frame->height) {
            dest_width = dest_frame->width;
            dest_height = hdest;
        } else {
            dest_width = wdest;
            dest_height = dest_frame->height;
        }

        switch (dest_frame->colorFormat) {
        case NVBUF_COLOR_FORMAT_RGBA:
            pixel_size = 4;
            break;
        case NVBUF_COLOR_FORMAT_RGB:
            pixel_size = 3;
            break;
        case NVBUF_COLOR_FORMAT_GRAY8:
        case NVBUF_COLOR_FORMAT_NV12:
            pixel_size = 1;
            break;
        default:
            g_assert_not_reached();
            break;
        }

        /* Pad the scaled image with black color. */
        cudaReturn = cudaMemset2DAsync((uint8_t *)destCudaPtr + pixel_size * dest_width,
                                       dest_frame->planeParams.pitch[0], 0,
                                       pixel_size * (dest_frame->width - dest_width),
                                       dest_frame->height, nvdspreprocess->convert_stream);
        if (cudaReturn != cudaSuccess) {
            GST_ERROR_OBJECT(nvdspreprocess,
                             "cudaMemset2DAsync failed with error %s while converting buffer",
                             cudaGetErrorName(cudaReturn));
            return GST_FLOW_ERROR;
        }
        cudaReturn = cudaMemset2DAsync(
            (uint8_t *)destCudaPtr + dest_frame->planeParams.pitch[0] * dest_height,
            dest_frame->planeParams.pitch[0], 0, pixel_size * dest_width,
            dest_frame->height - dest_height, nvdspreprocess->convert_stream);
        if (cudaReturn != cudaSuccess) {
            GST_ERROR_OBJECT(nvdspreprocess,
                             "cudaMemset2DAsync failed with error %s while converting buffer",
                             cudaGetErrorName(cudaReturn));
            return GST_FLOW_ERROR;
        }
    } else {
        GST_DEBUG_OBJECT(nvdspreprocess, "scaling at processing width & height\n");
        dest_width = nvdspreprocess->processing_width;
        dest_height = nvdspreprocess->processing_height;
    }

    /* Calculate the scaling ratio of the frame / object crop. This will be
     * required later for rescaling the detector output boxes to input resolution.
     */
    ratio_x = (double)dest_width / src_width;
    ratio_y = (double)dest_height / src_height;

    offset_left = dest_frame->width - dest_width;
    offset_top = dest_frame->height - dest_height;

#ifdef __aarch64__
    if (nvdspreprocess->scaling_pool_compute_hw != NvBufSurfTransformCompute_GPU) {
        if (ratio_x <= 1.0 / 16 || ratio_x >= 16.0) {
            GST_ELEMENT_ERROR(
                nvdspreprocess, STREAM, FAILED,
                ("%s:Currently cannot scale by ratio > 16 or < 1/16 for Jetson\n", __func__),
                (NULL));
            return GST_FLOW_ERROR;
        }
        if (ratio_y <= 1.0 / 16 || ratio_y >= 16.0) {
            GST_ELEMENT_ERROR(
                nvdspreprocess, STREAM, FAILED,
                ("%s:Currently cannot scale by ratio > 16 or < 1/16 for Jetson\n", __func__),
                (NULL));
            return GST_FLOW_ERROR;
        }
    }
#endif

    /* We will first convert only the Region of Interest (the entire frame or the
     * object bounding box) to RGB and then scale the converted RGB frame to
     * processing resolution. */
    GST_DEBUG_OBJECT(nvdspreprocess, "Fill the input buffer for batch transformation\n");

    /* Create temporary src and dest surfaces for NvBufSurfTransform API. */
    nvdspreprocess->batch_insurf.surfaceList[nvdspreprocess->batch_insurf.numFilled] = *src_frame;

    nvdspreprocess->batch_outsurf.surfaceList[nvdspreprocess->batch_outsurf.numFilled] =
        *dest_frame;

    /* Set the source ROI. Could be entire frame or an object. */
    nvdspreprocess->transform_params.src_rect[nvdspreprocess->batch_insurf.numFilled] = {
        (guint)src_top, (guint)src_left, (guint)src_width, (guint)src_height};
    /* Set the dest ROI. Could be the entire destination frame or part of it to
     * maintain aspect ratio. */
    nvdspreprocess->transform_params.dst_rect[nvdspreprocess->batch_outsurf.numFilled] = {
        0, 0, dest_width, dest_height};

    nvdspreprocess->batch_insurf.numFilled++;
    nvdspreprocess->batch_outsurf.numFilled++;

    nvdspreprocess->batch_insurf.batchSize = nvdspreprocess->batch_insurf.numFilled;
    nvdspreprocess->batch_outsurf.batchSize = nvdspreprocess->batch_outsurf.numFilled;

    return GST_FLOW_OK;
}

#ifdef DUMP_ROIS
static gboolean dump_rois(GstNvDsPreProcess *nvdspreprocess,
                          NvDsPreProcessBatch *batch,
                          NvBufSurface *outsurf)
{
    cv::Mat in_mat, out_mat;
    static guint cnt = 0;
    guint src_id = G_MAXUINT;
    guint roi_cnt = 0;

    for (guint i = 0; i < batch->units.size(); i++) {
        if (src_id == batch->units[i].frame_meta->source_id) {
            roi_cnt++;
        } else {
            roi_cnt = 0;
        }
        src_id = batch->units[i].frame_meta->source_id;

        // Map the buffer so that it can be accessed by CPU
        if (NvBufSurfaceMap(outsurf, i, 0, NVBUF_MAP_READ) != 0) {
            GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                              ("%s:buffer map to be accessed by CPU failed", __func__), (NULL));
            return FALSE;
        }

        // sync mapped data for CPU access
        NvBufSurfaceSyncForCpu(outsurf, i, 0);

        in_mat =
            cv::Mat(nvdspreprocess->processing_height, nvdspreprocess->processing_width, CV_8UC3,
                    outsurf->surfaceList[i].mappedAddr.addr[0], outsurf->surfaceList[i].pitch);

#ifdef __aarch64__
#if (CV_MAJOR_VERSION >= 4)
        cv::cvtColor(in_mat, out_mat, cv::COLOR_RGBA2BGR);
#else
        cv::cvtColor(in_mat, out_mat, CV_RGBA2BGR);
#endif
#else
        cv::cvtColor(in_mat, out_mat, cv::COLOR_RGB2BGR);
#endif
        cv::imwrite("out_" + std::to_string(cnt) + "__src__" + std::to_string(src_id) + "__roi__" +
                        std::to_string(roi_cnt) + ".jpeg",
                    out_mat);

        if (NvBufSurfaceUnMap(outsurf, i, 0)) {
            GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                              ("%s:buffer unmap to be accessed by CPU failed", __func__), (NULL));
            return FALSE;
        }

#ifdef __aarch64__
        // To use the converted buffer in CUDA, create an EGLImage and then use
        // CUDA-EGL interop APIs
        if (USE_EGLIMAGE) {
            if (NvBufSurfaceMapEglImage(outsurf, 0) != 0) {
                GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                                  ("%s:buffer map eglimage failed", __func__), (NULL));
                return FALSE;
            }
            // outsurf->surfaceList[0].mappedAddr.eglImage
            // Use interop APIs cuGraphicsEGLRegisterImage and
            // cuGraphicsResourceGetMappedEglFrame to access the buffer in CUDA

            // Destroy the EGLImage
            NvBufSurfaceUnMapEglImage(outsurf, 0);
        }
#endif
    }
    cnt++;

    return TRUE;
}
#endif

/** As an custom example we perform async batched transformation here. */
static gboolean batch_transformation(NvBufSurface *in_surf,
                                     NvBufSurface *out_surf,
                                     CustomTransformParams &params)
{
    NvBufSurfTransform_Error err;

    err = NvBufSurfTransformSetSessionParams(&params.transform_config_params);
    if (err != NvBufSurfTransformError_Success) {
        GST_ERROR("NvBufSurfTransformSetSessionParams failed with error %d\n", err);
        return FALSE;
    }

    /* Batched tranformation. */
    err = NvBufSurfTransformAsync(in_surf, out_surf, &params.transform_params, &params.sync_obj);

    if (err != NvBufSurfTransformError_Success) {
        GST_ERROR("NvBufSurfTransform failed with error %d\n", err);
        return FALSE;
    }

    return TRUE;
}

static gboolean group_transformation(GstNvDsPreProcess *nvdspreprocess,
                                     GstNvDsPreProcessGroup *&group)
{
    std::string nvtx_str;
    CustomTransformParams params;
    gboolean ret = 0;

    /** Configure transform session parameters for the transformation */
    params.transform_config_params = nvdspreprocess->transform_config_params;
    params.transform_params = nvdspreprocess->transform_params;

    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0xFFFF0000;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_str = "convert_buf batch_num=" + std::to_string(nvdspreprocess->current_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();

    nvtxDomainRangePushEx(nvdspreprocess->nvtx_domain, &eventAttrib);

    if (nvdspreprocess->custom_lib_path && nvdspreprocess->custom_lib_handle &&
        group->custom_transform_function_name) {
        NvDsPreProcessStatus status;
        status = group->custom_transform(&nvdspreprocess->batch_insurf,
                                         &nvdspreprocess->batch_outsurf, params);

        if (status != NVDSPREPROCESS_SUCCESS) {
            GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                              ("Custom Transformation from library failed\n"), (NULL));
            return FALSE;
        }
        ret = TRUE;
        GST_DEBUG_OBJECT(nvdspreprocess, "Custom tranformation from library successfull\n");
    } else {
        ret = batch_transformation(&nvdspreprocess->batch_insurf, &nvdspreprocess->batch_outsurf,
                                   params);
        if (!ret) {
            GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                              ("Transformation from plugin failed with error"), (NULL));
            return FALSE;
        }
        GST_DEBUG_OBJECT(nvdspreprocess, "Custom tranformation from plugin successfull\n");
    }

    group->sync_obj = params.sync_obj;

    nvtxDomainRangePop(nvdspreprocess->nvtx_domain);

    return ret;
}

static void release_user_meta_at_batch_level(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    GstNvDsPreProcessBatchMeta *preprocess_batchmeta =
        (GstNvDsPreProcessBatchMeta *)user_meta->user_meta_data;
    if (preprocess_batchmeta->tensor_meta != nullptr) {
        gst_buffer_unref((GstBuffer *)preprocess_batchmeta->tensor_meta
                             ->private_data); // unref tensor pool buffer
        delete preprocess_batchmeta->tensor_meta;
    }
    gst_buffer_unref(
        (GstBuffer *)preprocess_batchmeta->private_data); // unref conversion pool buffer

    for (auto &roi_meta : preprocess_batchmeta->roi_vector) {
        g_list_free(roi_meta.classifier_meta_list);
        g_list_free(roi_meta.roi_user_meta_list);
    }

    delete preprocess_batchmeta;
}

static void attach_user_meta_at_batch_level(GstNvDsPreProcess *nvdspreprocess,
                                            NvDsPreProcessBatch *batch,
                                            CustomTensorParams custom_tensor_params,
                                            NvDsPreProcessStatus status)
{
    NvDsBatchMeta *batch_meta = NULL;
    NvDsUserMeta *user_meta = NULL;

    /** attach preprocess batchmeta as user meta at batch level */
    GstNvDsPreProcessBatchMeta *preprocess_batchmeta = new GstNvDsPreProcessBatchMeta;

    /* If tensor is prepared */
    if (status == NVDSPREPROCESS_SUCCESS) {
        preprocess_batchmeta->roi_vector.clear();
        preprocess_batchmeta->roi_vector = custom_tensor_params.seq_params.roi_vector;

        preprocess_batchmeta->tensor_meta = new NvDsPreProcessTensorMeta;

        preprocess_batchmeta->tensor_meta->gpu_id = nvdspreprocess->gpu_id;
        preprocess_batchmeta->tensor_meta->private_data =
            ((NvDsPreProcessCustomBufImpl *)nvdspreprocess->tensor_buf)->gstbuf;
        preprocess_batchmeta->tensor_meta->raw_tensor_buffer =
            ((NvDsPreProcessCustomBufImpl *)nvdspreprocess->tensor_buf)->memory->dev_memory_ptr;
        preprocess_batchmeta->tensor_meta->tensor_shape =
            custom_tensor_params.params.network_input_shape;
        preprocess_batchmeta->tensor_meta->buffer_size = custom_tensor_params.params.buffer_size;
        preprocess_batchmeta->tensor_meta->data_type = custom_tensor_params.params.data_type;
        preprocess_batchmeta->tensor_meta->tensor_name = custom_tensor_params.params.tensor_name;

        GST_DEBUG_OBJECT(nvdspreprocess, "attached network shape in tensor %d : %d : %d : %d\n",
                         preprocess_batchmeta->tensor_meta->tensor_shape[0],
                         preprocess_batchmeta->tensor_meta->tensor_shape[1],
                         preprocess_batchmeta->tensor_meta->tensor_shape[2],
                         preprocess_batchmeta->tensor_meta->tensor_shape[3]);
    } else {
        preprocess_batchmeta->roi_vector.clear();
        for (guint i = 0; i < batch->units.size(); i++) {
            preprocess_batchmeta->roi_vector.push_back(batch->units[i].roi_meta);
        }
        preprocess_batchmeta->tensor_meta = nullptr; // set tensor meta to nullptr
    }

    preprocess_batchmeta->private_data = batch->converted_buf;
    preprocess_batchmeta->target_unique_ids = nvdspreprocess->target_unique_ids;

    batch_meta = batch->batch_meta;

    /* Acquire user meta from pool */
    user_meta = nvds_acquire_user_meta_from_pool(batch_meta);

    /* Set user meta below */
    user_meta->user_meta_data = preprocess_batchmeta;
    user_meta->base_meta.meta_type = (NvDsMetaType)NVDS_PREPROCESS_BATCH_META;
    user_meta->base_meta.copy_func = NULL;
    user_meta->base_meta.release_func = release_user_meta_at_batch_level;
    user_meta->base_meta.batch_meta = batch_meta;

    nvds_add_user_meta_to_batch(batch_meta, user_meta);

    return;
}

/* Process entire frames in the batched buffer. */
static GstFlowReturn gst_nvdspreprocess_on_frame(GstNvDsPreProcess *nvdspreprocess,
                                                 GstBuffer *inbuf,
                                                 NvBufSurface *in_surf)
{
    GstFlowReturn flow_ret = GST_FLOW_ERROR;
    std::string nvtx_str;
    std::unique_ptr<NvDsPreProcessBatch> batch = nullptr;

    GstNvDsPreProcessMemory *memory = nullptr;
    GstBuffer *conv_gst_buf = nullptr;

    NvDsBatchMeta *batch_meta = NULL;
    guint num_groups = 0;
    gdouble scale_ratio_x, scale_ratio_y;
    gdouble offset_left, offset_top;
    gint idx = 0;

    if (((in_surf->memType == NVBUF_MEM_DEFAULT || in_surf->memType == NVBUF_MEM_CUDA_DEVICE) &&
         ((int)in_surf->gpuId != (int)nvdspreprocess->gpu_id)) ||
        (((int)in_surf->gpuId == (int)nvdspreprocess->gpu_id) &&
         (in_surf->memType == NVBUF_MEM_SYSTEM))) {
        GST_ELEMENT_ERROR(nvdspreprocess, RESOURCE, FAILED,
                          ("Memory Compatibility Error:Input surface gpu-id doesnt match with "
                           "configured gpu-id for element,"
                           " please allocate input using unified memory, or use same gpu-ids OR,"
                           " if same gpu-ids are used ensure appropriate Cuda memories are used"),
                          ("surface-gpu-id=%d,%s-gpu-id=%d", in_surf->gpuId,
                           GST_ELEMENT_NAME(nvdspreprocess), nvdspreprocess->gpu_id));
        return GST_FLOW_ERROR;
    }

    batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);
    if (batch_meta == nullptr) {
        GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                          ("NvDsBatchMeta not found for input buffer."), (NULL));
        return GST_FLOW_ERROR;
    }

    if (batch == nullptr) {
        batch.reset(new NvDsPreProcessBatch);
        batch->push_buffer = FALSE;
        batch->inbuf = inbuf;
        batch->inbuf_batch_num = nvdspreprocess->current_batch_num;
        batch->batch_meta = batch_meta;
        batch->scaling_pool_format = nvdspreprocess->scaling_pool_format;

        flow_ret =
            gst_buffer_pool_acquire_buffer(nvdspreprocess->scaling_pool, &conv_gst_buf, nullptr);

        if (flow_ret != GST_FLOW_OK) {
            return flow_ret;
        }
        memory = gst_nvdspreprocess_buffer_get_memory(conv_gst_buf);
        if (!memory) {
            return GST_FLOW_ERROR;
        }
        batch->converted_buf = conv_gst_buf;
        batch->pitch = memory->surf->surfaceList[0].planeParams.pitch[0];
    }

    if (nvdspreprocess->nvdspreprocess_groups[0]->src_ids[0] == -1 &&
        batch_meta->num_frames_in_batch *
                nvdspreprocess->nvdspreprocess_groups[0]->framemeta_map[0].roi_vector.size() >
            nvdspreprocess->max_batch_size) {
        GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                          ("tensor shape batch-size should be atleast sum total rois\n"), (NULL));
        return GST_FLOW_ERROR;
    }

    num_groups = nvdspreprocess->nvdspreprocess_groups.size();
    GST_DEBUG_OBJECT(nvdspreprocess, "Num Groups = %d\n", num_groups);
    std::vector<bool> group_present(num_groups, 0);

    for (guint gcnt = 0; gcnt < num_groups; gcnt++) {
        GstNvDsPreProcessGroup *&preprocess_group = nvdspreprocess->nvdspreprocess_groups[gcnt];
        GST_DEBUG_OBJECT(nvdspreprocess, "num filled in batch meta = %d\n",
                         batch_meta->num_frames_in_batch);
        NvDsMetaList *l_frame = NULL;
        for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
            NvDsFrameMeta *frame_meta = NULL;
            frame_meta = (NvDsFrameMeta *)(l_frame->data);

            gint source_id = frame_meta->source_id;  /* source id of incoming buffer */
            gint batch_index = frame_meta->batch_id; /* batch id of incoming buffer */

            GstNvDsPreProcessFrame preprocess_frame;
            std::vector<NvDsRoiMeta> roi_vector;
            NvDsRoiMeta roi_meta;
            NvOSD_RectParams rect_params;

            std::vector<gint> src_ids = preprocess_group->src_ids;

            if (src_ids[0] == -1) {
                num_groups = 1;
            }

            if (std::find(src_ids.begin(), src_ids.end(), source_id) == src_ids.end() &&
                src_ids[0] != -1) {
                GST_DEBUG_OBJECT(nvdspreprocess, "Group %d : No Source %d => skipping\n", gcnt,
                                 source_id);
                continue;
            }

            GST_DEBUG_OBJECT(nvdspreprocess, "Group %d : Processsing Source ID = %d \n", gcnt,
                             source_id);

            auto get_preprocess_frame_meta = preprocess_group->framemeta_map.find(source_id);

            if (get_preprocess_frame_meta == preprocess_group->framemeta_map.end() &&
                src_ids[0] != -1) {
                g_print("Group %d : Configuration for Source ID = %d not found\n", gcnt, source_id);
                flow_ret = GST_FLOW_ERROR;
                return flow_ret;
            } else {
                preprocess_frame = get_preprocess_frame_meta->second;

                roi_vector = preprocess_frame.roi_vector;

                GST_DEBUG_OBJECT(nvdspreprocess,
                                 "Group %d : Source ID %d : Got roi-vecsize = %ld\n", gcnt,
                                 source_id, roi_vector.size());

                for (guint n = 0; n < preprocess_frame.roi_vector.size(); n++) {
                    roi_meta = roi_vector[n];

                    if (preprocess_group->process_on_roi) {
                        GST_DEBUG_OBJECT(nvdspreprocess,
                                         "Group %d : Source ID %d : Processing on ROIS\n", gcnt,
                                         source_id);

                        /** Process on ROIs provided from config file */
                        rect_params = roi_meta.roi;

                        GST_DEBUG_OBJECT(nvdspreprocess,
                                         "filling ROI left=%f top=%f width=%f height=%f\n",
                                         rect_params.left, rect_params.top, rect_params.width,
                                         rect_params.height);
                    } else {
                        GST_DEBUG_OBJECT(nvdspreprocess,
                                         "Group %d : Source ID %d : Processing on Full Frames\n",
                                         gcnt, source_id);

                        /** Process on Full Frames */
                        rect_params.left = 0;
                        rect_params.top = 0;
                        rect_params.width = in_surf->surfaceList[batch_index].width;
                        rect_params.height = in_surf->surfaceList[batch_index].height;

                        roi_meta.roi = rect_params;
                        GST_DEBUG_OBJECT(nvdspreprocess,
                                         "filling FULL FRAME left=%f top=%f width=%f height=%f\n",
                                         rect_params.left, rect_params.top, rect_params.width,
                                         rect_params.height);
                    }

#ifdef DRAW_ROIS
                    NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                    display_meta->num_rects = 1;
                    display_meta->rect_params[0].left = rect_params.left;
                    display_meta->rect_params[0].top = rect_params.top;
                    display_meta->rect_params[0].width = rect_params.width;
                    display_meta->rect_params[0].height = rect_params.height;
                    display_meta->rect_params[0].border_width = 2;
                    display_meta->rect_params[0].border_color = {0, 1, 0, 1};

                    nvds_add_display_meta_to_frame(frame_meta, display_meta);
#endif

                    idx = batch->units.size();

                    if (idx == nvdspreprocess->tensor_params.network_input_shape[0]) {
                        GST_ELEMENT_WARNING(nvdspreprocess, STREAM, FAILED,
                                            ("Warning: exceeding preprocess batch-size "
                                             "(=network-input-shape[0]) => skipping sources\n"),
                                            (NULL));
                        continue;
                    }

                    /** Scale the roi part to the network resolution maintaining aspect ratio */
                    if (scale_and_fill_data(nvdspreprocess, in_surf->surfaceList + batch_index,
                                            &rect_params, scale_ratio_x, scale_ratio_y, offset_left,
                                            offset_top, memory->surf,
                                            memory->surf->surfaceList + idx,
                                            memory->frame_memory_ptrs[idx]) != GST_FLOW_OK) {
                        flow_ret = GST_FLOW_ERROR;
                        return flow_ret;
                    }

                    roi_meta.converted_buffer =
                        (NvBufSurfaceParams *)memory->surf->surfaceList + idx;
                    roi_meta.scale_ratio_x = scale_ratio_x;
                    roi_meta.scale_ratio_y = scale_ratio_y;
                    roi_meta.offset_left = offset_left;
                    roi_meta.offset_top = offset_top;
                    roi_meta.frame_meta = frame_meta;

                    /* Adding a Unit (ROI/Crop/Full Frame) to the current batch. Set the frames
                     * members. */
                    NvDsPreProcessUnit unit;
                    unit.converted_frame_ptr = memory->frame_memory_ptrs[idx];
                    unit.obj_meta = nullptr;
                    unit.frame_meta = frame_meta;
                    unit.frame_num = unit.frame_meta->frame_num;
                    unit.batch_index = batch_index;
                    unit.input_surf_params = in_surf->surfaceList + batch_index;
                    unit.roi_meta = roi_meta;
                    unit.roi_meta.classifier_meta_list = NULL;
                    unit.roi_meta.roi_user_meta_list = NULL;

                    batch->units.push_back(unit);

                    if (preprocess_group->process_on_roi) {
                        GST_DEBUG_OBJECT(nvdspreprocess,
                                         "Group %d : Source ID %d : ROI : max-batch-size = %d "
                                         "batch-units-size = %ld batch_index = %d idx = %d\n",
                                         gcnt, source_id, nvdspreprocess->max_batch_size,
                                         batch->units.size(), batch_index, idx);
                    } else {
                        GST_DEBUG_OBJECT(nvdspreprocess,
                                         "Group %d : Source ID %d : FULL FRAME : max-batch-size = "
                                         "%d batch-units-size = %ld batch_index = %d idx = %d\n",
                                         gcnt, source_id, nvdspreprocess->max_batch_size,
                                         batch->units.size(), batch_index, idx);
                    }
                }
            }
        }
        /** transform the group according to num filled from batch_meta */
        if (nvdspreprocess->batch_insurf.numFilled < preprocess_group->num_units) {
            GST_DEBUG_OBJECT(nvdspreprocess, "Some Sources not received for Group %d\n", gcnt);
        }
        if (nvdspreprocess->batch_insurf.numFilled > 0) {
            if (!group_transformation(nvdspreprocess, preprocess_group)) {
                GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                                  ("Group %d : group transformation failed\n", gcnt), (NULL));
                return GST_FLOW_ERROR;
            }
            group_present[gcnt] = 1;
        } else {
            GST_DEBUG_OBJECT(nvdspreprocess, "No Sources received for Group %d\n", gcnt);
        }
        nvdspreprocess->batch_insurf.numFilled = 0;
        nvdspreprocess->batch_outsurf.numFilled = 0;
    }

#ifdef DUMP_ROIS
    gboolean ret = dump_rois(nvdspreprocess, batch.get(), memory->surf);
    if (!ret) {
        g_print("dump_rois failed\n");
        return GST_FLOW_ERROR;
    }
#endif

    /** wait for async transformation */
    for (guint gcnt = 0; gcnt < num_groups; gcnt++) {
        GstNvDsPreProcessGroup *x = nvdspreprocess->nvdspreprocess_groups[gcnt];
        if (x->custom_transform_function_name == NULL ||
            !g_strcmp0(x->custom_transform_function_name, "CustomAsyncTransformation")) {
            if (group_present[gcnt] == 1) {
                NvBufSurfTransformSyncObjWait(x->sync_obj, -1);
                NvBufSurfTransformSyncObjDestroy(&x->sync_obj);
            }
        }
    }

    /* Push the batch info structure in the processing queue and notify the process
     * thread that a new batch has been queued. */
    g_mutex_lock(&nvdspreprocess->preprocess_lock);

    g_queue_push_tail(nvdspreprocess->preprocess_queue, batch.get());
    g_cond_broadcast(&nvdspreprocess->preprocess_cond);

    g_mutex_unlock(&nvdspreprocess->preprocess_lock);

    /* Batch submitted. Set batch to nullptr so that a new NvDsPreProcessBatch
     * structure can be allocated if required. */
    conv_gst_buf = nullptr;
    batch.release();

    return GST_FLOW_OK;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn gst_nvdspreprocess_submit_input_buffer(GstBaseTransform *btrans,
                                                            gboolean discont,
                                                            GstBuffer *inbuf)
{
    GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS(btrans);
    GstMapInfo in_map_info;
    NvBufSurface *in_surf;
    NvDsPreProcessBatch *buf_push_batch;
    GstFlowReturn flow_ret = GST_FLOW_ERROR;
    std::string nvtx_str;
    std::unique_ptr<NvDsPreProcessBatch> batch = nullptr;

    nvdspreprocess->current_batch_num++;

    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0xFFFF0000;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_str = "buffer_process batch_num=" + std::to_string(nvdspreprocess->current_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();
    nvtxRangeId_t buf_process_range =
        nvtxDomainRangeStartEx(nvdspreprocess->nvtx_domain, &eventAttrib);

    if (FALSE == nvdspreprocess->config_file_parse_successful) {
        GST_ELEMENT_ERROR(nvdspreprocess, LIBRARY, SETTINGS,
                          ("Configuration file parsing failed\n"),
                          ("Config file path: %s\n", nvdspreprocess->config_file_path));
        return flow_ret;
    }

    if (FALSE == nvdspreprocess->enable) {
        GST_DEBUG_OBJECT(nvdspreprocess, "nvdspreprocess in passthrough mode\n");
        flow_ret = gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(nvdspreprocess), inbuf);
        return flow_ret;
    }

    memset(&in_map_info, 0, sizeof(in_map_info));

    /* Map the buffer contents and get the pointer to NvBufSurface. */
    if (!gst_buffer_map(inbuf, &in_map_info, GST_MAP_READ)) {
        GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                          ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__),
                          (NULL));
        return GST_FLOW_ERROR;
    }
    in_surf = (NvBufSurface *)in_map_info.data;

    nvds_set_input_system_timestamp(inbuf, GST_ELEMENT_NAME(nvdspreprocess));

    /** Preprocess on Frames */
    flow_ret = gst_nvdspreprocess_on_frame(nvdspreprocess, inbuf, in_surf);

    if (flow_ret != GST_FLOW_OK)
        goto error;

    nvtxDomainRangeEnd(nvdspreprocess->nvtx_domain, buf_process_range);

    /* Queue a push buffer batch. This batch is not inferred. This batch is to
     * signal the process thread that there are no more batches
     * belonging to this input buffer and this GstBuffer can be pushed to
     * downstream element once all the previous processing is done. */
    buf_push_batch = new NvDsPreProcessBatch;
    buf_push_batch->inbuf = inbuf;
    buf_push_batch->push_buffer = TRUE;
    buf_push_batch->nvtx_complete_buf_range = buf_process_range;

    g_mutex_lock(&nvdspreprocess->preprocess_lock);
    /* Check if this is a push buffer or event marker batch. If yes, no need to
     * queue the input for inferencing. */
    if (buf_push_batch->push_buffer) {
        /* Push the batch info structure in the processing queue and notify the
         * process thread that a new batch has been queued. */
        g_queue_push_tail(nvdspreprocess->preprocess_queue, buf_push_batch);
        g_cond_broadcast(&nvdspreprocess->preprocess_cond);
    }
    g_mutex_unlock(&nvdspreprocess->preprocess_lock);

    flow_ret = GST_FLOW_OK;

error:
    gst_buffer_unmap(inbuf, &in_map_info);
    return flow_ret;
}

/**
 * If submit_input_buffer is implemented, it is mandatory to implement
 * generate_output. Buffers are not pushed to the downstream element from here.
 * Return the GstFlowReturn value of the latest pad push so that any error might
 * be caught by the application.
 */
static GstFlowReturn gst_nvdspreprocess_generate_output(GstBaseTransform *btrans,
                                                        GstBuffer **outbuf)
{
    GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS(btrans);
    return nvdspreprocess->last_flow_ret;
}

NvDsPreProcessAcquirerImpl::NvDsPreProcessAcquirerImpl(GstBufferPool *pool)
{
    m_gstpool = pool;
}

NvDsPreProcessCustomBuf *NvDsPreProcessAcquirerImpl::acquire()
{
    GstBuffer *gstbuf;
    GstNvDsPreProcessMemory *memory;
    GstFlowReturn flow_ret;

    flow_ret = gst_buffer_pool_acquire_buffer(m_gstpool, &gstbuf, nullptr);

    if (flow_ret != GST_FLOW_OK) {
        GST_ERROR("error while acquiring buffer from tensor pool\n");
        return nullptr;
    }

    memory = gst_nvdspreprocess_buffer_get_memory(gstbuf);
    if (!memory) {
        GST_ERROR("error while getting memory from tensor pool\n");
        return nullptr;
    }

    return new NvDsPreProcessCustomBufImpl{{memory->dev_memory_ptr}, gstbuf, memory};
}

gboolean NvDsPreProcessAcquirerImpl::release(NvDsPreProcessCustomBuf *buf)
{
    NvDsPreProcessCustomBufImpl *implBuf = (NvDsPreProcessCustomBufImpl *)(buf);
    gst_buffer_unref(implBuf->gstbuf);
    delete implBuf;
    return TRUE;
}

/**
 * Output loop used to pop output from processing thread, attach the output to the
 * buffer in form of NvDsMeta and push the buffer to downstream element.
 */
static gpointer gst_nvdspreprocess_output_loop(gpointer data)
{
    GstNvDsPreProcess *nvdspreprocess = GST_NVDSPREPROCESS(data);
    NvDsPreProcessStatus status = NVDSPREPROCESS_TENSOR_NOT_READY;

    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0xFFFF0000;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    std::string nvtx_str;
    nvtx_str = "gst-nvdspreprocess_output-loop_uid=" + std::to_string(nvdspreprocess->unique_id);

    g_mutex_lock(&nvdspreprocess->preprocess_lock);

    /* Run till signalled to stop. */
    while (!nvdspreprocess->stop) {
        std::unique_ptr<NvDsPreProcessBatch> batch = nullptr;

        /* Wait if processing queue is empty. */
        if (g_queue_is_empty(nvdspreprocess->preprocess_queue)) {
            g_cond_wait(&nvdspreprocess->preprocess_cond, &nvdspreprocess->preprocess_lock);
            continue;
        }

        /* Pop a batch from the element's process queue. */
        batch.reset((NvDsPreProcessBatch *)g_queue_pop_head(nvdspreprocess->preprocess_queue));
        g_cond_broadcast(&nvdspreprocess->preprocess_cond);

        /* Event marker used for synchronization. No need to process further. */
        if (batch->event_marker) {
            continue;
        }

        g_mutex_unlock(&nvdspreprocess->preprocess_lock);

        /* Need to only push buffer to downstream element. This batch was not
         * actually submitted for inferencing. */
        if (batch->push_buffer) {
            nvtxDomainRangeEnd(nvdspreprocess->nvtx_domain, batch->nvtx_complete_buf_range);

            nvds_set_output_system_timestamp(batch->inbuf, GST_ELEMENT_NAME(nvdspreprocess));

            GstFlowReturn flow_ret =
                gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(nvdspreprocess), batch->inbuf);
            if (nvdspreprocess->last_flow_ret != flow_ret) {
                switch (flow_ret) {
                    /* Signal the application for pad push errors by posting a error message
                     * on the pipeline bus. */
                case GST_FLOW_ERROR:
                case GST_FLOW_NOT_LINKED:
                case GST_FLOW_NOT_NEGOTIATED:
                    GST_ELEMENT_ERROR(nvdspreprocess, STREAM, FAILED,
                                      ("Internal data stream error."),
                                      ("streaming stopped, reason %s (%d)",
                                       gst_flow_get_name(flow_ret), flow_ret));
                    break;
                default:
                    break;
                }
            }
            nvdspreprocess->last_flow_ret = flow_ret;
            g_mutex_lock(&nvdspreprocess->preprocess_lock);
            continue;
        }

        nvtx_str = "dequeueOutputAndAttachMeta batch_num=" + std::to_string(batch->inbuf_batch_num);
        eventAttrib.message.ascii = nvtx_str.c_str();
        nvtxDomainRangePushEx(nvdspreprocess->nvtx_domain, &eventAttrib);

        CustomTensorParams custom_tensor_params;
        /** Tensor Preparation from Custom Library */
        if (nvdspreprocess->custom_lib_path && nvdspreprocess->custom_lib_handle &&
            nvdspreprocess->custom_tensor_function) {
            custom_tensor_params.params = nvdspreprocess->tensor_params;
            custom_tensor_params.seq_params.roi_vector.clear();

            for (guint i = 0; i < batch->units.size(); i++) {
                custom_tensor_params.seq_params.roi_vector.push_back(batch->units[i].roi_meta);
            }

            status = nvdspreprocess->custom_tensor_function(
                nvdspreprocess->custom_lib_ctx, batch.get(), nvdspreprocess->tensor_buf,
                custom_tensor_params, nvdspreprocess->acquire_impl.get());
        }

#ifdef DEBUG_TENSOR
        static int batch_num = 0;
        std::ofstream outfile("tensorout_batch_" + std::to_string(batch_num) + ".bin");
        outfile.write((char *)((NvDsPreProcessCustomBufImpl *)nvdspreprocess->tensor_buf)
                          ->memory->dev_memory_ptr,
                      custom_tensor_params.params.buffer_size);
        outfile.close();
        batch_num++;
#endif

        /** attach user meta at batch level */
        attach_user_meta_at_batch_level(nvdspreprocess, batch.get(), custom_tensor_params, status);

        g_mutex_lock(&nvdspreprocess->preprocess_lock);

        nvtxDomainRangePop(nvdspreprocess->nvtx_domain);
    }
    g_mutex_unlock(&nvdspreprocess->preprocess_lock);

    return nullptr;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean nvdspreprocess_plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvdspreprocess_debug, "nvdspreprocess", 0, "preprocess plugin");

    return gst_element_register(plugin, "nvdspreprocess", GST_RANK_PRIMARY,
                                GST_TYPE_NVDSPREPROCESS);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  nvdsgst_preprocess,
                  DESCRIPTION,
                  nvdspreprocess_plugin_init,
                  "6.0",
                  LICENSE,
                  BINARY_PACKAGE,
                  URL)
