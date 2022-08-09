/**
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "gstdsexample.h"

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

GST_DEBUG_CATEGORY_STATIC(gst_dsexample_debug);
#define GST_CAT_DEFAULT gst_dsexample_debug
#define USE_EGLIMAGE 1

#define NVDS_IMG_CROP_OBJECT_USER_OBJECT_META                               \
    nvds_get_user_meta_type(((gchar *)"NVIDIA.IMG.CROP.OBJECT.USER.OBJECT." \
                                      "META"))

#define NVDS_MASK_CROP_OBJECT_USER_OBJECT_META                               \
    nvds_get_user_meta_type(((gchar *)"NVIDIA.MASK.CROP.OBJECT.USER.OBJECT." \
                                      "META"))

/* enable to write transformed cvmat to files */
/* #define DSEXAMPLE_DEBUG */
static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum {
    PROP_0,
    PROP_UNIQUE_ID,
    PROP_PROCESSING_WIDTH,
    PROP_PROCESSING_HEIGHT,
    PROP_PROCESS_FULL_FRAME,
    PROP_GPU_DEVICE_ID
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
#define DEFAULT_PROCESS_FULL_FRAME TRUE
#define DEFAULT_GPU_ID 0

// TODO: Move to property
#define PATH_SAVE ""
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
static GstStaticPadTemplate gst_dsexample_sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_dsexample_src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_dsexample_parent_class parent_class
G_DEFINE_TYPE(GstDsExample, gst_dsexample, GST_TYPE_BASE_TRANSFORM);

static void gst_dsexample_set_property(GObject *object,
                                       guint prop_id,
                                       const GValue *value,
                                       GParamSpec *pspec);
static void gst_dsexample_get_property(GObject *object,
                                       guint prop_id,
                                       GValue *value,
                                       GParamSpec *pspec);

static gboolean gst_dsexample_set_caps(GstBaseTransform *btrans, GstCaps *incaps, GstCaps *outcaps);
static gboolean gst_dsexample_start(GstBaseTransform *btrans);
static gboolean gst_dsexample_stop(GstBaseTransform *btrans);

static GstFlowReturn gst_dsexample_transform_ip(GstBaseTransform *btrans, GstBuffer *inbuf);

static void attach_metadata_object(GstDsExample *dsexample,
                                   NvDsObjectMeta *obj_meta,
                                   gchar *file_path,
                                   NvDsMetaType user_meta_type);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void gst_dsexample_class_init(GstDsExampleClass *klass)
{
    GObjectClass *gobject_class;
    GstElementClass *gstelement_class;
    GstBaseTransformClass *gstbasetransform_class;

    /* Indicates we want to use DS buf api */
    g_setenv("DS_NEW_BUFAPI", "1", TRUE);

    gobject_class = (GObjectClass *)klass;
    gstelement_class = (GstElementClass *)klass;
    gstbasetransform_class = (GstBaseTransformClass *)klass;

    /* Overide base class functions */
    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_dsexample_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_dsexample_get_property);

    gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR(gst_dsexample_set_caps);
    gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_dsexample_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_dsexample_stop);

    gstbasetransform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_dsexample_transform_ip);

    /* Install properties */
    g_object_class_install_property(
        gobject_class, PROP_UNIQUE_ID,
        g_param_spec_uint("unique-id", "Unique ID",
                          "Unique ID for the element. Can be used to identify output of the"
                          " element",
                          0, G_MAXUINT, DEFAULT_UNIQUE_ID,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_PROCESSING_WIDTH,
        g_param_spec_int("processing-width", "Processing Width",
                         "Width of the input buffer to algorithm", 1, G_MAXINT,
                         DEFAULT_PROCESSING_WIDTH,
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_PROCESSING_HEIGHT,
        g_param_spec_int("processing-height", "Processing Height",
                         "Height of the input buffer to algorithm", 1, G_MAXINT,
                         DEFAULT_PROCESSING_HEIGHT,
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_GPU_DEVICE_ID,
        g_param_spec_uint(
            "gpu-id", "Set GPU Device ID", "Set GPU Device ID", 0, G_MAXUINT, 0,
            GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    /* Set sink and src pad capabilities */
    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&gst_dsexample_src_template));
    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&gst_dsexample_sink_template));

    /* Set metadata describing the element */
    gst_element_class_set_details_simple(
        gstelement_class, "DsExample plugin", "DsExample Plugin",
        "Process a 3rdparty example algorithm on objects / full frame",
        "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
        "@ https://devtalk.nvidia.com/default/board/209/");
}

static void gst_dsexample_init(GstDsExample *dsexample)
{
    GstBaseTransform *btrans = GST_BASE_TRANSFORM(dsexample);

    /* We will not be generating a new buffer. Just adding / updating
     * metadata. */
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(btrans), TRUE);
    /* We do not want to change the input caps. Set to passthrough. transform_ip
     * is still called. */
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(btrans), TRUE);

    /* Initialize all property variables to default values */
    dsexample->unique_id = DEFAULT_UNIQUE_ID;
    dsexample->processing_width = DEFAULT_PROCESSING_WIDTH;
    dsexample->processing_height = DEFAULT_PROCESSING_HEIGHT;
    dsexample->gpu_id = DEFAULT_GPU_ID;

    /* This quark is required to identify NvDsMeta when iterating through
     * the buffer metadatas */
    if (!_dsmeta_quark)
        _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void gst_dsexample_set_property(GObject *object,
                                       guint prop_id,
                                       const GValue *value,
                                       GParamSpec *pspec)
{
    GstDsExample *dsexample = GST_DSEXAMPLE(object);
    switch (prop_id) {
    case PROP_UNIQUE_ID:
        dsexample->unique_id = g_value_get_uint(value);
        break;
    case PROP_PROCESSING_WIDTH:
        dsexample->processing_width = g_value_get_int(value);
        break;
    case PROP_PROCESSING_HEIGHT:
        dsexample->processing_height = g_value_get_int(value);
        break;
    case PROP_GPU_DEVICE_ID:
        dsexample->gpu_id = g_value_get_uint(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void gst_dsexample_get_property(GObject *object,
                                       guint prop_id,
                                       GValue *value,
                                       GParamSpec *pspec)
{
    GstDsExample *dsexample = GST_DSEXAMPLE(object);

    switch (prop_id) {
    case PROP_UNIQUE_ID:
        g_value_set_uint(value, dsexample->unique_id);
        break;
    case PROP_PROCESSING_WIDTH:
        g_value_set_int(value, dsexample->processing_width);
        break;
    case PROP_PROCESSING_HEIGHT:
        g_value_set_int(value, dsexample->processing_height);
        break;
    case PROP_GPU_DEVICE_ID:
        g_value_set_uint(value, dsexample->gpu_id);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean gst_dsexample_start(GstBaseTransform *btrans)
{
    GstDsExample *dsexample = GST_DSEXAMPLE(btrans);
    NvBufSurfaceCreateParams create_params;

    GstQuery *queryparams = NULL;
    guint batch_size = 1;
    int val = -1;

    /* Algorithm specific initializations and resource allocation. */

    CHECK_CUDA_STATUS(cudaSetDevice(dsexample->gpu_id), "Unable to set cuda device");

    cudaDeviceGetAttribute(&val, cudaDevAttrIntegrated, dsexample->gpu_id);
    dsexample->is_integrated = val;

    dsexample->batch_size = 1;
    queryparams = gst_nvquery_batch_size_new();
    if (gst_pad_peer_query(GST_BASE_TRANSFORM_SINK_PAD(btrans), queryparams) ||
        gst_pad_peer_query(GST_BASE_TRANSFORM_SRC_PAD(btrans), queryparams)) {
        if (gst_nvquery_batch_size_parse(queryparams, &batch_size)) {
            dsexample->batch_size = batch_size;
        }
    }
    GST_DEBUG_OBJECT(dsexample, "Setting batch-size %d \n", dsexample->batch_size);
    gst_query_unref(queryparams);

    CHECK_CUDA_STATUS(cudaStreamCreate(&dsexample->cuda_stream), "Could not create cuda stream");

    if (dsexample->inter_buf)
        NvBufSurfaceDestroy(dsexample->inter_buf);
    dsexample->inter_buf = NULL;

    /* An intermediate buffer for NV12/RGBA to BGR conversion  will be
     * required. Can be skipped if custom algorithm can work directly on NV12/RGBA.
     */
    create_params.gpuId = dsexample->gpu_id;
    create_params.width = dsexample->processing_width;
    create_params.height = dsexample->processing_height;
    create_params.size = 0;
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    create_params.layout = NVBUF_LAYOUT_PITCH;

    if (dsexample->is_integrated) {
        create_params.memType = NVBUF_MEM_DEFAULT;
    } else {
        create_params.memType = NVBUF_MEM_CUDA_PINNED;
    }

    if (NvBufSurfaceCreate(&dsexample->inter_buf, 1, &create_params) != 0) {
        GST_ERROR("Error: Could not allocate internal buffer for dsexample");
        goto error;
    }

    /* Create host memory for storing converted/scaled interleaved RGB data */
    CHECK_CUDA_STATUS(cudaMallocHost(&dsexample->host_rgb_buf, dsexample->processing_width *
                                                                   dsexample->processing_height *
                                                                   RGB_BYTES_PER_PIXEL),
                      "Could not allocate cuda host buffer");

    GST_DEBUG_OBJECT(dsexample, "allocated cuda buffer %p \n", dsexample->host_rgb_buf);

    return TRUE;
error:
    if (dsexample->host_rgb_buf) {
        cudaFreeHost(dsexample->host_rgb_buf);
        dsexample->host_rgb_buf = NULL;
    }

    if (dsexample->cuda_stream) {
        cudaStreamDestroy(dsexample->cuda_stream);
        dsexample->cuda_stream = NULL;
    }
    return FALSE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean gst_dsexample_stop(GstBaseTransform *btrans)
{
    GstDsExample *dsexample = GST_DSEXAMPLE(btrans);

    if (dsexample->inter_buf)
        NvBufSurfaceDestroy(dsexample->inter_buf);
    dsexample->inter_buf = NULL;

    if (dsexample->cuda_stream)
        cudaStreamDestroy(dsexample->cuda_stream);
    dsexample->cuda_stream = NULL;

    if (dsexample->host_rgb_buf) {
        cudaFreeHost(dsexample->host_rgb_buf);
        dsexample->host_rgb_buf = NULL;
    }

    GST_DEBUG_OBJECT(dsexample, "deleted CV Mat \n");

    return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 * Get the negotiated caps parameters and process them accordingly.
 * Such as obtaining the width and height information of the input image
 *
 * Gets the capabilities of the video (i.e. resolution, color format, framerate) that flow through
 * this element. Allocations / initializations that depend on input video format can be done here.
 */
static gboolean gst_dsexample_set_caps(GstBaseTransform *btrans, GstCaps *incaps, GstCaps *outcaps)
{
    GstDsExample *dsexample = GST_DSEXAMPLE(btrans);
    /* Save the input video information, since this will be required later. */
    gst_video_info_from_caps(&dsexample->video_info, incaps);

    return TRUE;
}

#ifdef WITH_OPENCV
static void resizeMask(float *src,
                       int original_width,
                       int original_height,
                       cv::Mat &dst,
                       float threshold)
{
    auto clip = [](float in, float low, float high) -> float {
        return (in < low) ? low : (in > high ? high : in);
    };
    auto target_height = dst.rows;
    auto target_width = dst.cols;
    float ratio_h = static_cast<float>(original_height) / static_cast<float>(target_height);
    float ratio_w = static_cast<float>(original_width) / static_cast<float>(target_width);
    int channel = 1;

    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            float x0 = static_cast<float>(x) * ratio_w;
            float y0 = static_cast<float>(y) * ratio_h;
            int left = static_cast<int>(
                clip(std::floor(x0), 0.0f, static_cast<float>(original_width - 1.0f)));
            int top = static_cast<int>(
                clip(std::floor(y0), 0.0f, static_cast<float>(original_height - 1.0f)));
            int right = static_cast<int>(
                clip(std::ceil(x0), 0.0f, static_cast<float>(original_width - 1.0f)));
            int bottom = static_cast<int>(
                clip(std::ceil(y0), 0.0f, static_cast<float>(original_height - 1.0f)));

            for (int c = 0; c < channel; ++c) {
                // H, W, C ordering
                float left_top_val =
                    (float)src[top * (original_width * channel) + left * (channel) + c];
                float right_top_val =
                    (float)src[top * (original_width * channel) + right * (channel) + c];
                float left_bottom_val =
                    (float)src[bottom * (original_width * channel) + left * (channel) + c];
                float right_bottom_val =
                    (float)src[bottom * (original_width * channel) + right * (channel) + c];
                float top_lerp = left_top_val + (right_top_val - left_top_val) * (x0 - left);
                float bottom_lerp =
                    left_bottom_val + (right_bottom_val - left_bottom_val) * (x0 - left);
                float lerp = top_lerp + (bottom_lerp - top_lerp) * (y0 - top);
                if (lerp < threshold) {
                    dst.at<unsigned char>(y, x) = 0;
                } else {
                    dst.at<unsigned char>(y, x) = 255;
                }
            }
        }
    }
}
#endif

/**
 * Scale the entire frame to the processing resolution maintaining aspect ratio.
 * Or crop and scale objects to the processing resolution maintaining the aspect
 * ratio. Remove the padding required by hardware and convert from RGBA to RGB
 * using openCV. These steps can be skipped if the algorithm can work with
 * padded data and/or can work with RGBA.
 *
 * Scales, converts, or crops the input buffer, either the full frame or the
 * object based on its co-ordinates in primary detector metadata
 */
static GstFlowReturn get_converted_mat(GstDsExample *dsexample,
                                       NvBufSurface *input_buf,
                                       gint idx,
                                       NvOSD_RectParams *crop_rect_params,
                                       NvOSD_MaskParams *mask_params,
                                       gdouble &ratio,
                                       gint input_width,
                                       gint input_height,
                                       char *img_file_path,
                                       char *mask_file_path)
{
    NvBufSurfTransform_Error err;
    NvBufSurfTransformConfigParams transform_config_params;
    NvBufSurfTransformParams transform_params;
    NvBufSurfTransformRect src_rect;
    NvBufSurfTransformRect dst_rect;
    NvBufSurface ip_surf;
#ifdef WITH_OPENCV
    cv::Mat in_mat;
    cv::Mat out_mat;
    cv::Mat dst_mask;
#endif
    ip_surf = *input_buf;

    ip_surf.numFilled = ip_surf.batchSize = 1;
    ip_surf.surfaceList = &(input_buf->surfaceList[idx]);

    gint src_left = GST_ROUND_UP_2((unsigned int)crop_rect_params->left);
    gint src_top = GST_ROUND_UP_2((unsigned int)crop_rect_params->top);
    gint src_width = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->width);
    gint src_height = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->height);
    guint dest_width = src_width, dest_height = src_height;

    NvBufSurfaceCreateParams create_params;
    create_params.gpuId = dsexample->gpu_id;
    create_params.width = dest_width;
    create_params.height = dest_height;
    create_params.size = 0;
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    create_params.layout = NVBUF_LAYOUT_PITCH;

    if (dsexample->is_integrated) {
        create_params.memType = NVBUF_MEM_DEFAULT;
    } else {
        create_params.memType = NVBUF_MEM_CUDA_PINNED;
    }

    if (NvBufSurfaceCreate(&dsexample->inter_buf, 1, &create_params) != 0) {
        GST_ERROR("Error: Could not allocate internal buffer for dsexample");
        goto error;
    }

    /* Configure transform session parameters for the transformation */
    transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
    transform_config_params.gpu_id = dsexample->gpu_id;
    transform_config_params.cuda_stream = dsexample->cuda_stream;

    /* Set the transform session parameters for the conversions executed in this
     * thread.
     */
    err = NvBufSurfTransformSetSessionParams(&transform_config_params);
    if (err != NvBufSurfTransformError_Success) {
        GST_ELEMENT_ERROR(dsexample, STREAM, FAILED,
                          ("NvBufSurfTransformSetSessionParams failed with error %d", err), (NULL));
        goto error;
    }

    /* Calculate scaling ratio while maintaining aspect ratio */
    ratio = MIN(1.0 * dest_width / src_width, 1.0 * dest_height / src_height);

    if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0)) {
        GST_ELEMENT_ERROR(dsexample, STREAM, FAILED,
                          ("%s:crop_rect_params dimensions are zero", __func__), (NULL));
        goto error;
    }

#ifdef __aarch64__
    if (ratio <= 1.0 / 16 || ratio >= 16.0) {
        /* Currently cannot scale by ratio > 16 or < 1/16 for Jetson */
        goto error;
    }
#endif
    /* Set the transform ROIs for source and destination */
    src_rect = {(guint)src_top, (guint)src_left, (guint)src_width, (guint)src_height};
    dst_rect = {0, 0, (guint)dest_width, (guint)dest_height};

    /* Set the transform parameters */
    transform_params.src_rect = &src_rect;
    transform_params.dst_rect = &dst_rect;
    transform_params.transform_flag =
        NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC | NVBUFSURF_TRANSFORM_CROP_DST;
    transform_params.transform_filter = NvBufSurfTransformInter_Default;

    /* Memset the memory */
    NvBufSurfaceMemSet(dsexample->inter_buf, 0, 0, 0);

    GST_DEBUG_OBJECT(dsexample, "Scaling and converting input buffer\n");

    /* Transformation scaling+format conversion if any. */
    err = NvBufSurfTransform(&ip_surf, dsexample->inter_buf, &transform_params);
    if (err != NvBufSurfTransformError_Success) {
        GST_ELEMENT_ERROR(dsexample, STREAM, FAILED,
                          ("NvBufSurfTransform failed with error %d while converting buffer", err),
                          (NULL));
        goto error;
    }
    /* Map the buffer so that it can be accessed by CPU */
    if (NvBufSurfaceMap(dsexample->inter_buf, 0, 0, NVBUF_MAP_READ) != 0) {
        goto error;
    }
    if (dsexample->inter_buf->memType == NVBUF_MEM_SURFACE_ARRAY) {
        /* Cache the mapped data for CPU access */
        NvBufSurfaceSyncForCpu(dsexample->inter_buf, 0, 0);
    }

#ifdef WITH_OPENCV
    /* Use openCV to remove padding and convert RGBA to BGR. Can be skipped if
     * algorithm can handle padded RGBA data. */
    in_mat = cv::Mat(dest_height, dest_width, CV_8UC4,
                     dsexample->inter_buf->surfaceList[0].mappedAddr.addr[0],
                     dsexample->inter_buf->surfaceList[0].pitch);
    out_mat = cv::Mat(cv::Size(dest_width, dest_height), CV_8UC3);
    dst_mask = cv::Mat(dest_height, dest_width, CV_8UC1);

#if (CV_MAJOR_VERSION >= 4)
    cv::cvtColor(in_mat, out_mat, cv::COLOR_RGBA2BGR);
#else
    cv::cvtColor(in_mat, out_mat, CV_RGBA2BGR);
#endif

    resizeMask(mask_params->data, mask_params->width, mask_params->height, dst_mask,
               mask_params->threshold);

    cv::imwrite(img_file_path, out_mat);
    cv::imwrite(mask_file_path, dst_mask);
#endif

    if (NvBufSurfaceUnMap(dsexample->inter_buf, 0, 0)) {
        goto error;
    }

    if (dsexample->is_integrated) {
#ifdef __aarch64__
        /* To use the converted buffer in CUDA, create an EGLImage and then use
         * CUDA-EGL interop APIs */
        if (USE_EGLIMAGE) {
            if (NvBufSurfaceMapEglImage(dsexample->inter_buf, 0) != 0) {
                goto error;
            }

            /* dsexample->inter_buf->surfaceList[0].mappedAddr.eglImage
             * Use interop APIs cuGraphicsEGLRegisterImage and
             * cuGraphicsResourceGetMappedEglFrame to access the buffer in CUDA */

            /* Destroy the EGLImage */
            NvBufSurfaceUnMapEglImage(dsexample->inter_buf, 0);
        }
#endif
    }

    /* We will first convert only the Region of Interest (the entire frame or the
     * object bounding box) to RGB and then scale the converted RGB frame to
     * processing resolution. */
    return GST_FLOW_OK;

error:
    return GST_FLOW_ERROR;
}

/**
 * In pass-through mode, after obtaining upstream data and performing algorithm processing,
 * the detected results are superimposed in GstBuffer and passed to the downstream
 *
 * Called when element recieves an input buffer from upstream element.
 *
 * Implemented in the simple version. Called when the plugin receives a buffer from upstream element
 *  - Finds the metadata of the primary detector
 *  - Use get_converted_mat to pre-process frame/object crop to get the required buffer for pushing
 * to library. Push the data to the example library. Pop the example library output.
 *  - Attach / update metadata using attach_metadata_full_frame or attach_metadata_object.
 *  - Alternatively, modify frame contents in-place to blur objects using blur_objects.
 */
static GstFlowReturn gst_dsexample_transform_ip(GstBaseTransform *btrans, GstBuffer *inbuf)
{
    GstDsExample *dsexample = GST_DSEXAMPLE(btrans);
    GstMapInfo in_map_info;
    GstFlowReturn flow_ret = GST_FLOW_ERROR;
    gdouble scale_ratio = 1.0;
    // DsExampleOutput *output;

    NvBufSurface *surface = NULL;
    NvDsBatchMeta *batch_meta = NULL;
    NvDsFrameMeta *frame_meta = NULL;
    NvDsMetaList *l_frame = NULL;
    // guint i = 0;

    /* Using object crops as input to the algorithm. The objects are detected by
     * the primary detector
     */
    NvDsMetaList *l_obj = NULL;
    NvDsObjectMeta *obj_meta = NULL;

    dsexample->frame_num++;
    CHECK_CUDA_STATUS(cudaSetDevice(dsexample->gpu_id), "Unable to set cuda device");

    memset(&in_map_info, 0, sizeof(in_map_info));
    if (!gst_buffer_map(inbuf, &in_map_info, GST_MAP_READ)) {
        g_print("Error: Failed to map gst buffer\n");
        goto error;
    }

    nvds_set_input_system_timestamp(inbuf, GST_ELEMENT_NAME(dsexample));
    surface = (NvBufSurface *)in_map_info.data;
    GST_DEBUG_OBJECT(dsexample, "Processing Frame %" G_GUINT64_FORMAT " Surface %p\n",
                     dsexample->frame_num, surface);

    if (CHECK_NVDS_MEMORY_AND_GPUID(dsexample, surface))
        goto error;

    batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);
    if (batch_meta == nullptr) {
        GST_ELEMENT_ERROR(dsexample, STREAM, FAILED, ("NvDsBatchMeta not found for input buffer."),
                          (NULL));
        return GST_FLOW_ERROR;
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta *)(l_frame->data);
        NvOSD_RectParams rect_params;

        /* Scale the entire frame to processing resolution */
        rect_params.left = 0;
        rect_params.top = 0;
        rect_params.width = dsexample->video_info.width;
        rect_params.height = dsexample->video_info.height;

        guint id_cropobj = 0;

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            id_cropobj++;

            obj_meta = (NvDsObjectMeta *)(l_obj->data);

            if (obj_meta->unique_component_id != 1)
                continue;

            /* Should not process on objects smaller than MIN_INPUT_OBJECT_WIDTH x
             * MIN_INPUT_OBJECT_HEIGHT since it will cause hardware scaling issues.
             */
            if ((unsigned int)obj_meta->rect_params.width < MIN_INPUT_OBJECT_WIDTH ||
                (unsigned int)obj_meta->rect_params.height < MIN_INPUT_OBJECT_HEIGHT)
                continue;

            // system(g_strdup_printf("mkdir -p %s/%d", PATH_SAVE, obj_meta->object_id));
            // char *file_path = g_strdup_printf("%s/%d/image_%d_%d_%d.jpeg", PATH_SAVE,
            // obj_meta->object_id, frame_meta->frame_num, obj_meta->unique_component_id,
            // id_cropobj);

            if (!obj_meta->mask_params.data || obj_meta->mask_params.size <= 0)
                continue;

            gchar *img_file_path =
                g_strdup_printf("%s/image_%d_%d_%d.png", PATH_SAVE, frame_meta->frame_num,
                                obj_meta->unique_component_id, id_cropobj);

            gchar *mask_file_path =
                g_strdup_printf("%s/mask_%d_%d_%d.png", PATH_SAVE, frame_meta->frame_num,
                                obj_meta->unique_component_id, id_cropobj);

            /* Crop and scale the object */
            if (get_converted_mat(dsexample, surface, frame_meta->batch_id, &obj_meta->rect_params,
                                  &obj_meta->mask_params, scale_ratio, dsexample->video_info.width,
                                  dsexample->video_info.height, img_file_path,
                                  mask_file_path) != GST_FLOW_OK) {
                /* Error in conversion, skip processing on object. */
                continue;
            }

            attach_metadata_object(dsexample, obj_meta, img_file_path,
                                   NVDS_IMG_CROP_OBJECT_USER_OBJECT_META);

            attach_metadata_object(dsexample, obj_meta, mask_file_path,
                                   NVDS_MASK_CROP_OBJECT_USER_OBJECT_META);
        }
    }

    flow_ret = GST_FLOW_OK;

error:
    nvds_set_output_system_timestamp(inbuf, GST_ELEMENT_NAME(dsexample));
    gst_buffer_unmap(inbuf, &in_map_info);
    return flow_ret;
}

/* copy function set by user. "data" holds a pointer to NvDsUserMeta*/
static gpointer copy_ds_crop_object_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    return (gpointer)(g_strdup((gchar *)user_meta->user_meta_data));
}

/* release function set by user. "data" holds a pointer to NvDsUserMeta*/
static void release_ds_crop_object_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;

    if (user_meta->user_meta_data) {
        g_free(user_meta->user_meta_data);
        user_meta->user_meta_data = NULL;
    }
}

/**
 * Only update string label in an existing object metadata. No bounding boxes.
 * We assume only one label per object is generated
 */
static void attach_metadata_object(GstDsExample *dsexample,
                                   NvDsObjectMeta *obj_meta,
                                   gchar *file_path,
                                   NvDsMetaType user_meta_type)
{
    NvDsBatchMeta *batch_meta = obj_meta->base_meta.batch_meta;

    // Attach - DsDirection MetaData
    NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);

    user_meta->user_meta_data = (gpointer)file_path;
    user_meta->base_meta.meta_type = user_meta_type;
    user_meta->base_meta.copy_func = copy_ds_crop_object_meta;
    user_meta->base_meta.release_func = release_ds_crop_object_meta;

    nvds_add_user_meta_to_obj(obj_meta, user_meta);
}

/**
 * Used to register plugins with GStreamer
 *
 * Boiler plate for registering a plugin and an element.
 */
static gboolean dsexample_plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_dsexample_debug, "dsexample", 0, "dsexample plugin");

    return gst_element_register(plugin, "dsexample", GST_RANK_PRIMARY, GST_TYPE_DSEXAMPLE);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  nvdsgst_dsexample,
                  DESCRIPTION,
                  dsexample_plugin_init,
                  "6.0",
                  LICENSE,
                  BINARY_PACKAGE,
                  URL)
