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

#include "gstdspostprocessing.h"

#include <string.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <sstream>
#include <string>

#include "nvdsmeta_schema.h"

GST_DEBUG_CATEGORY_STATIC(gst_dspostprocessing_debug);
#define GST_CAT_DEFAULT gst_dspostprocessing_debug
#define USE_EGLIMAGE 1

#define NVDS_COLOR_POST_PROCESSING2_USER_OBJECT_CENTER_META \
    nvds_get_user_meta_type(((gchar *)"NVIDIA.COLOR.POST.PROCESSING2.USER.OBJECT.CENTER.META"))

#define NVDS_COLOR_POST_PROCESSING2_USER_OBJECT_COUTER_META \
    nvds_get_user_meta_type(((gchar *)"NVIDIA.COLOR.POST.PROCESSING2.USER.OBJECT.COUTER.META"))

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min, max) (MAX(MIN(a, max), min))

/* enable to write transformed cvmat to files */
/* #define DSPOSTPROCESSING_DEBUG */
static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum { PROP_0, PROP_UNIQUE_ID, PROP_GPU_DEVICE_ID };

#define CHECK_NVDS_MEMORY_AND_GPUID(object, surface)                                  \
    ({                                                                                \
        int _errtype = 0;                                                             \
        do {                                                                          \
            if ((surface->memType == NVBUF_MEM_DEFAULT ||                             \
                 surface->memType == NVBUF_MEM_CUDA_DEVICE) &&                        \
                (surface->gpuId != object->gpu_id)) {                                 \
                GST_ELEMENT_ERROR(object, RESOURCE, FAILED,                           \
                                  ("Input surface gpu-id doesnt match with "          \
                                   "configured gpu-id for element,"                   \
                                   " please allocate input using unified memory, or " \
                                   "use same gpu-ids"),                               \
                                  ("surface-gpu-id=%d,%s-gpu-id=%d", surface->gpuId,  \
                                   GST_ELEMENT_NAME(object), object->gpu_id));        \
                _errtype = 1;                                                         \
            }                                                                         \
        } while (0);                                                                  \
        _errtype;                                                                     \
    })

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_GPU_ID 0

// TODO: Move to property
#define PATH_SAVE "./outputs/images"

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
static GstStaticPadTemplate gst_dspostprocessing_sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_dspostprocessing_src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_dspostprocessing_parent_class parent_class
G_DEFINE_TYPE(GstDsPostProcessing, gst_dspostprocessing, GST_TYPE_BASE_TRANSFORM);

static void gst_dspostprocessing_set_property(GObject *object,
                                              guint prop_id,
                                              const GValue *value,
                                              GParamSpec *pspec);
static void gst_dspostprocessing_get_property(GObject *object,
                                              guint prop_id,
                                              GValue *value,
                                              GParamSpec *pspec);

static gboolean gst_dspostprocessing_set_caps(GstBaseTransform *btrans,
                                              GstCaps *incaps,
                                              GstCaps *outcaps);
static gboolean gst_dspostprocessing_start(GstBaseTransform *btrans);
static gboolean gst_dspostprocessing_stop(GstBaseTransform *btrans);

static GstFlowReturn gst_dspostprocessing_transform_ip(GstBaseTransform *btrans, GstBuffer *inbuf);

static void attach_metadata_full_frame(GstDsPostProcessing *dspostprocessing,
                                       NvDsFrameMeta *frame_meta,
                                       gdouble scale_ratio,
                                       guint batch_id);
static void attach_center_object(GstDsPostProcessing *dspostprocessing,
                                 NvDsObjectMeta *obj_meta,
                                 int **centers);
static void attach_couter_object(GstDsPostProcessing *dspostprocessing,
                                 NvDsObjectMeta *obj_meta,
                                 int *couters);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void gst_dspostprocessing_class_init(GstDsPostProcessingClass *klass)
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
    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_dspostprocessing_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_dspostprocessing_get_property);

    gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR(gst_dspostprocessing_set_caps);
    gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_dspostprocessing_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_dspostprocessing_stop);

    gstbasetransform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_dspostprocessing_transform_ip);

    /* Install properties */
    g_object_class_install_property(
        gobject_class, PROP_UNIQUE_ID,
        g_param_spec_uint("unique-id", "Unique ID",
                          "Unique ID for the element. Can be used to identify output of the"
                          " element",
                          0, G_MAXUINT, DEFAULT_UNIQUE_ID,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_GPU_DEVICE_ID,
        g_param_spec_uint(
            "gpu-id", "Set GPU Device ID", "Set GPU Device ID", 0, G_MAXUINT, 0,
            GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));
    /* Set sink and src pad capabilities */
    gst_element_class_add_pad_template(
        gstelement_class, gst_static_pad_template_get(&gst_dspostprocessing_src_template));
    gst_element_class_add_pad_template(
        gstelement_class, gst_static_pad_template_get(&gst_dspostprocessing_sink_template));

    /* Set metadata describing the element */
    gst_element_class_set_details_simple(
        gstelement_class, "DsPostProcessing plugin", "DsPostProcessing Plugin",
        "Process a 3rdparty postprocessing algorithm on objects / full frame",
        "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
        "@ https://devtalk.nvidia.com/default/board/209/");
}

static void gst_dspostprocessing_init(GstDsPostProcessing *dspostprocessing)
{
    GstBaseTransform *btrans = GST_BASE_TRANSFORM(dspostprocessing);

    /* We will not be generating a new buffer. Just adding / updating
     * metadata. */
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(btrans), TRUE);
    /* We do not want to change the input caps. Set to passthrough. transform_ip
     * is still called. */
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(btrans), TRUE);

    /* Initialize all property variables to default values */
    dspostprocessing->unique_id = DEFAULT_UNIQUE_ID;
    dspostprocessing->gpu_id = DEFAULT_GPU_ID;

    /* This quark is required to identify NvDsMeta when iterating through
     * the buffer metadatas */
    if (!_dsmeta_quark)
        _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void gst_dspostprocessing_set_property(GObject *object,
                                              guint prop_id,
                                              const GValue *value,
                                              GParamSpec *pspec)
{
    GstDsPostProcessing *dspostprocessing = GST_DSPOSTPROCESSING(object);
    switch (prop_id) {
    case PROP_UNIQUE_ID:
        dspostprocessing->unique_id = g_value_get_uint(value);
        break;
    case PROP_GPU_DEVICE_ID:
        dspostprocessing->gpu_id = g_value_get_uint(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void gst_dspostprocessing_get_property(GObject *object,
                                              guint prop_id,
                                              GValue *value,
                                              GParamSpec *pspec)
{
    GstDsPostProcessing *dspostprocessing = GST_DSPOSTPROCESSING(object);

    switch (prop_id) {
    case PROP_UNIQUE_ID:
        g_value_set_uint(value, dspostprocessing->unique_id);
        break;
    case PROP_GPU_DEVICE_ID:
        g_value_set_uint(value, dspostprocessing->gpu_id);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean gst_dspostprocessing_start(GstBaseTransform *btrans)
{
    GstDsPostProcessing *dspostprocessing = GST_DSPOSTPROCESSING(btrans);
    NvBufSurfaceCreateParams create_params;

    GstQuery *queryparams = NULL;
    guint batch_size = 1;
    int val = -1;

    CHECK_CUDA_STATUS(cudaSetDevice(dspostprocessing->gpu_id), "Unable to set cuda device");

    cudaDeviceGetAttribute(&val, cudaDevAttrIntegrated, dspostprocessing->gpu_id);
    dspostprocessing->is_integrated = val;

    dspostprocessing->batch_size = 1;
    queryparams = gst_nvquery_batch_size_new();
    if (gst_pad_peer_query(GST_BASE_TRANSFORM_SINK_PAD(btrans), queryparams) ||
        gst_pad_peer_query(GST_BASE_TRANSFORM_SRC_PAD(btrans), queryparams)) {
        if (gst_nvquery_batch_size_parse(queryparams, &batch_size)) {
            dspostprocessing->batch_size = batch_size;
        }
    }
    GST_DEBUG_OBJECT(dspostprocessing, "Setting batch-size %d \n", dspostprocessing->batch_size);
    gst_query_unref(queryparams);

#ifndef WITH_OPENCV
    GST_ELEMENT_ERROR(dspostprocessing, STREAM, FAILED,
                      ("OpenCV has been deprecated, hence object blurring will not work."
                       "Enable OpenCV compilation in gst-dspostprocessing Makefile by setting "
                       "'WITH_OPENCV:=1"),
                      (NULL));
    goto error;
#endif

    CHECK_CUDA_STATUS(cudaStreamCreate(&dspostprocessing->cuda_stream),
                      "Could not create cuda stream");

    if (dspostprocessing->inter_buf)
        NvBufSurfaceDestroy(dspostprocessing->inter_buf);
    dspostprocessing->inter_buf = NULL;

    return TRUE;
error:
    if (dspostprocessing->host_rgb_buf) {
        cudaFreeHost(dspostprocessing->host_rgb_buf);
        dspostprocessing->host_rgb_buf = NULL;
    }

    if (dspostprocessing->cuda_stream) {
        cudaStreamDestroy(dspostprocessing->cuda_stream);
        dspostprocessing->cuda_stream = NULL;
    }
    return FALSE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean gst_dspostprocessing_stop(GstBaseTransform *btrans)
{
    GstDsPostProcessing *dspostprocessing = GST_DSPOSTPROCESSING(btrans);

    if (dspostprocessing->inter_buf)
        NvBufSurfaceDestroy(dspostprocessing->inter_buf);
    dspostprocessing->inter_buf = NULL;

    if (dspostprocessing->cuda_stream)
        cudaStreamDestroy(dspostprocessing->cuda_stream);
    dspostprocessing->cuda_stream = NULL;

    if (dspostprocessing->host_rgb_buf) {
        cudaFreeHost(dspostprocessing->host_rgb_buf);
        dspostprocessing->host_rgb_buf = NULL;
    }

    GST_DEBUG_OBJECT(dspostprocessing, "deleted CV Mat \n");

    GST_DEBUG_OBJECT(dspostprocessing, "ctx lib released \n");

    return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean gst_dspostprocessing_set_caps(GstBaseTransform *btrans,
                                              GstCaps *incaps,
                                              GstCaps *outcaps)
{
    GstDsPostProcessing *dspostprocessing = GST_DSPOSTPROCESSING(btrans);
    /* Save the input video information, since this will be required later. */
    gst_video_info_from_caps(&dspostprocessing->video_info, incaps);

    /* requires RGBA format for blurring the objects in opencv */
    if (dspostprocessing->video_info.finfo->format != GST_VIDEO_FORMAT_RGBA) {
        GST_ELEMENT_ERROR(dspostprocessing, STREAM, FAILED,
                          ("input format should be RGBA when using blur-objects property"), (NULL));
        goto error;
    }

    return TRUE;

error:
    return FALSE;
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

cv::Vec3b lab2rgb(cv::Vec3b input)
{
    cv::Mat lab(1, 1, CV_8UC3, cv::Scalar(input[0], input[1], input[2]));

    cv::Mat rgb_image;
    cv::cvtColor(lab, rgb_image, cv::COLOR_Lab2RGB);

    return rgb_image.at<cv::Vec3b>(0, 0);
}
#endif

/**
 * Scale the entire frame to the processing resolution maintaining aspect ratio.
 * Or crop and scale objects to the processing resolution maintaining the aspect
 * ratio. Remove the padding required by hardware and convert from RGBA to RGB
 * using openCV. These steps can be skipped if the algorithm can work with
 * padded data and/or can work with RGBA.
 */
static GstFlowReturn get_converted_mat(GstDsPostProcessing *dspostprocessing,
                                       NvBufSurface *input_buf,
                                       gint idx,
                                       NvOSD_RectParams *crop_rect_params,
                                       NvOSD_MaskParams *mask_params,
                                       gdouble &ratio,
                                       gint input_width,
                                       gint input_height,
                                       int **&output_centers,
                                       int *&output_couters)
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
#if (CV_MAJOR_VERSION < 4)
    cv::Mat out_mat_rgb;
#endif

    cv::Mat centers, labels, dst, output_image, output_image2;
    int K = 5;
    std::vector<cv::Vec3b> points;
    int num_pixels;

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
    create_params.gpuId = dspostprocessing->gpu_id;
    create_params.width = dest_width;
    create_params.height = dest_height;
    create_params.size = 0;
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    create_params.layout = NVBUF_LAYOUT_PITCH;

    if (dspostprocessing->is_integrated) {
        create_params.memType = NVBUF_MEM_DEFAULT;
    } else {
        create_params.memType = NVBUF_MEM_CUDA_PINNED;
    }

    if (NvBufSurfaceCreate(&dspostprocessing->inter_buf, 1, &create_params) != 0) {
        GST_ERROR("Error: Could not allocate internal buffer for dspostprocessing");
        goto error;
    }

    /* Configure transform session parameters for the transformation */
    transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
    transform_config_params.gpu_id = dspostprocessing->gpu_id;
    transform_config_params.cuda_stream = dspostprocessing->cuda_stream;

    /* Set the transform session parameters for the conversions executed in this
     * thread. */
    err = NvBufSurfTransformSetSessionParams(&transform_config_params);
    if (err != NvBufSurfTransformError_Success) {
        GST_ELEMENT_ERROR(dspostprocessing, STREAM, FAILED,
                          ("NvBufSurfTransformSetSessionParams failed with error %d", err), (NULL));
        goto error;
    }

    /* Calculate scaling ratio while maintaining aspect ratio */
    ratio = MIN(1.0 * dest_width / src_width, 1.0 * dest_height / src_height);

    if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0)) {
        GST_ELEMENT_ERROR(dspostprocessing, STREAM, FAILED,
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
    NvBufSurfaceMemSet(dspostprocessing->inter_buf, 0, 0, 0);

    GST_DEBUG_OBJECT(dspostprocessing, "Scaling and converting input buffer\n");

    /* Transformation scaling+format conversion if any. */
    err = NvBufSurfTransform(&ip_surf, dspostprocessing->inter_buf, &transform_params);
    if (err != NvBufSurfTransformError_Success) {
        GST_ELEMENT_ERROR(dspostprocessing, STREAM, FAILED,
                          ("NvBufSurfTransform failed with error %d while converting buffer", err),
                          (NULL));
        goto error;
    }
    /* Map the buffer so that it can be accessed by CPU */
    if (NvBufSurfaceMap(dspostprocessing->inter_buf, 0, 0, NVBUF_MAP_READ) != 0) {
        goto error;
    }
    if (dspostprocessing->inter_buf->memType == NVBUF_MEM_SURFACE_ARRAY) {
        /* Cache the mapped data for CPU access */
        NvBufSurfaceSyncForCpu(dspostprocessing->inter_buf, 0, 0);
    }

#ifdef WITH_OPENCV
    /* Use openCV to remove padding and convert RGBA to BGR. Can be skipped if
     * algorithm can handle padded RGBA data. */
    in_mat = cv::Mat(dest_height, dest_width, CV_8UC4,
                     dspostprocessing->inter_buf->surfaceList[0].mappedAddr.addr[0],
                     dspostprocessing->inter_buf->surfaceList[0].pitch);
    out_mat = cv::Mat(cv::Size(dest_width, dest_height), CV_8UC3);

#if (CV_MAJOR_VERSION >= 4)
    cv::cvtColor(in_mat, out_mat, cv::COLOR_RGBA2Lab);
#else
    cv::cvtColor(in_mat, out_mat_rgb, CV_RGBA2RGB);
    cv::cvtColor(out_mat_rgb, out_mat, CV_RGB2Lab);
#endif
    dst = cv::Mat(dest_height, dest_width, CV_8UC1);

    resizeMask(mask_params->data, mask_params->width, mask_params->height, dst,
               mask_params->threshold);

    num_pixels = 0;
    for (int row_index = 0; row_index < dst.rows; row_index++) {
        for (int col_index = 0; col_index < dst.cols; col_index++) {
            if (dst.at<unsigned char>(row_index, col_index) == 255) {
                num_pixels += 1;
            }
        }
    }

    output_image = cv::Mat(num_pixels, 3, CV_32F);

    num_pixels = 0;
    for (int row_index = 0; row_index < dst.rows; row_index++) {
        for (int col_index = 0; col_index < dst.cols; col_index++) {
            if (dst.at<unsigned char>(row_index, col_index) == 255) {
                cv::Vec3b lab = out_mat.at<cv::Vec3b>(cv::Point(row_index, col_index));
                output_image.at<float>(num_pixels, 0) = lab[0] / 255.0;
                output_image.at<float>(num_pixels, 1) = lab[1] / 255.0;
                output_image.at<float>(num_pixels, 2) = lab[2] / 255.0;
                num_pixels += 1;
            }
        }
    }

    cv::kmeans(output_image, K, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 50, 0.0001), 5,
               cv::KMEANS_PP_CENTERS, centers);

    output_centers = new int *[centers.rows];
    output_couters = new int[centers.rows]{0};

    for (int i = 0; i < labels.rows; i++) {
        output_couters[labels.at<int>(i)]++;
    }

    for (int i = 0; i < centers.rows; i++) {
        cv::Vec3b rgb = lab2rgb(cv::Vec3b({
            (int)CLIP(centers.at<float>(i, 0) * 255.0, 0, 255),
            (int)CLIP(centers.at<float>(i, 1) * 255.0, 0, 255),
            (int)CLIP(centers.at<float>(i, 2) * 255.0, 0, 255),
        }));

        *(output_centers + i) = new int[3]{rgb[0], rgb[1], rgb[2]};
    }
#endif

    if (NvBufSurfaceUnMap(dspostprocessing->inter_buf, 0, 0)) {
        goto error;
    }

    if (dspostprocessing->is_integrated) {
#ifdef __aarch64__
        /* To use the converted buffer in CUDA, create an EGLImage and then use
         * CUDA-EGL interop APIs */
        if (USE_EGLIMAGE) {
            if (NvBufSurfaceMapEglImage(dspostprocessing->inter_buf, 0) != 0) {
                goto error;
            }

            /* dspostprocessing->inter_buf->surfaceList[0].mappedAddr.eglImage
             * Use interop APIs cuGraphicsEGLRegisterImage and
             * cuGraphicsResourceGetMappedEglFrame to access the buffer in CUDA */

            /* Destroy the EGLImage */
            NvBufSurfaceUnMapEglImage(dspostprocessing->inter_buf, 0);
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
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn gst_dspostprocessing_transform_ip(GstBaseTransform *btrans, GstBuffer *inbuf)
{
    GstDsPostProcessing *dspostprocessing = GST_DSPOSTPROCESSING(btrans);
    GstMapInfo in_map_info;
    GstFlowReturn flow_ret = GST_FLOW_ERROR;
    gdouble scale_ratio = 1.0;

    NvBufSurface *surface = NULL;
    NvDsBatchMeta *batch_meta = NULL;
    NvDsFrameMeta *frame_meta = NULL;
    NvDsMetaList *l_frame = NULL;
    guint i = 0;

    /* Using object crops as input to the algorithm. The objects are detected by
     * the primary detector */
    NvDsMetaList *l_obj = NULL;
    NvDsObjectMeta *obj_meta = NULL;

    dspostprocessing->frame_num++;
    CHECK_CUDA_STATUS(cudaSetDevice(dspostprocessing->gpu_id), "Unable to set cuda device");

    memset(&in_map_info, 0, sizeof(in_map_info));
    if (!gst_buffer_map(inbuf, &in_map_info, GST_MAP_READ)) {
        g_print("Error: Failed to map gst buffer\n");
        goto error;
    }

    nvds_set_input_system_timestamp(inbuf, GST_ELEMENT_NAME(dspostprocessing));
    surface = (NvBufSurface *)in_map_info.data;
    GST_DEBUG_OBJECT(dspostprocessing, "Processing Frame %" G_GUINT64_FORMAT " Surface %p\n",
                     dspostprocessing->frame_num, surface);

    if (CHECK_NVDS_MEMORY_AND_GPUID(dspostprocessing, surface))
        goto error;

    batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);
    if (batch_meta == nullptr) {
        GST_ELEMENT_ERROR(dspostprocessing, STREAM, FAILED,
                          ("NvDsBatchMeta not found for input buffer."), (NULL));
        return GST_FLOW_ERROR;
    }

    if (!dspostprocessing->is_integrated) {
        if (!(surface->memType == NVBUF_MEM_CUDA_UNIFIED ||
              surface->memType == NVBUF_MEM_CUDA_PINNED)) {
            GST_ELEMENT_ERROR(dspostprocessing, STREAM, FAILED,
                              ("%s:need NVBUF_MEM_CUDA_UNIFIED or NVBUF_MEM_CUDA_PINNED "
                               "memory for opencv blurring",
                               __func__),
                              (NULL));
            return GST_FLOW_ERROR;
        }
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta *)(l_frame->data);

        guint id_cropobj = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            id_cropobj++;

            obj_meta = (NvDsObjectMeta *)(l_obj->data);

            /* Should not process on objects smaller than MIN_INPUT_OBJECT_WIDTH x
             * MIN_INPUT_OBJECT_HEIGHT since it will cause hardware scaling issues. */
            if (obj_meta->rect_params.width < MIN_INPUT_OBJECT_WIDTH ||
                obj_meta->rect_params.height < MIN_INPUT_OBJECT_HEIGHT)
                continue;

            if (!obj_meta->mask_params.data || obj_meta->mask_params.size <= 0)
                continue;

            int **centers;
            int *couters;

            /* Crop and scale the object */
            if (get_converted_mat(
                    dspostprocessing, surface, frame_meta->batch_id, &obj_meta->rect_params,
                    &obj_meta->mask_params, scale_ratio, dspostprocessing->video_info.width,
                    dspostprocessing->video_info.height, centers, couters) != GST_FLOW_OK) {
                /* Error in conversion, skip processing on object. */
                continue;
            }

            /* Attach labels for the object */
            attach_center_object(dspostprocessing, obj_meta, centers);
            attach_couter_object(dspostprocessing, obj_meta, couters);
        }
    }

    flow_ret = GST_FLOW_OK;

error:

    nvds_set_output_system_timestamp(inbuf, GST_ELEMENT_NAME(dspostprocessing));
    gst_buffer_unmap(inbuf, &in_map_info);
    return flow_ret;
}

/**
 * Attach metadata for the full frame. We will be adding a new metadata.
 */
static void attach_metadata_full_frame(GstDsPostProcessing *dspostprocessing,
                                       NvDsFrameMeta *frame_meta,
                                       gdouble scale_ratio,
                                       guint batch_id)
{
}

/* copy function set by user. "data" holds a pointer to NvDsUserMeta*/
static gpointer object_couter_copy_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsCustomMsgInfo *srcMeta = (NvDsCustomMsgInfo *)user_meta->user_meta_data;
    NvDsCustomMsgInfo *dstMeta =
        (NvDsCustomMsgInfo *)g_memdup((gpointer)srcMeta, sizeof(NvDsCustomMsgInfo));

    dstMeta->message = (void *)g_memdup((void *)srcMeta->message, srcMeta->size * sizeof(int));
    dstMeta->size = srcMeta->size;

    return dstMeta;
}

/* release function set by user. "data" holds a pointer to NvDsUserMeta*/
static void object_couter_free_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsCustomMsgInfo *srcMeta = (NvDsCustomMsgInfo *)user_meta->user_meta_data;
    user_meta->user_meta_data = NULL;

    if (srcMeta->message) {
        g_free(srcMeta->message);
    }
    g_free(srcMeta);
}

/**
 * Only update string label in an existing object metadata. No bounding boxes.
 * We assume only one label per object is generated
 */
static void attach_couter_object(GstDsPostProcessing *dspostprocessing,
                                 NvDsObjectMeta *obj_meta,
                                 int *couters)
{
    NvDsBatchMeta *batch_meta = obj_meta->base_meta.batch_meta;

    // Attach - DsDirection MetaData
    NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    NvDsMetaType user_meta_type = NVDS_COLOR_POST_PROCESSING2_USER_OBJECT_COUTER_META;

    NvDsCustomMsgInfo *custom_msg_info = (NvDsCustomMsgInfo *)g_malloc0(sizeof(NvDsCustomMsgInfo));
    custom_msg_info->message = (void *)couters;
    custom_msg_info->size = 5; // TODO

    user_meta->user_meta_data = custom_msg_info;
    user_meta->base_meta.meta_type = user_meta_type;
    user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)object_couter_copy_func;
    user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)object_couter_free_func;

    nvds_add_user_meta_to_obj(obj_meta, user_meta);
}

/* copy function set by user. "data" holds a pointer to NvDsUserMeta*/
static gpointer object_center_copy_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsCustomMsgInfo *srcMeta = (NvDsCustomMsgInfo *)user_meta->user_meta_data;
    NvDsCustomMsgInfo *dstMeta =
        (NvDsCustomMsgInfo *)g_memdup((gpointer)srcMeta, sizeof(NvDsCustomMsgInfo));

    int **src_centers = (int **)srcMeta->message;
    int **dst_centers = new int *[srcMeta->size];

    for (int i = 0; i < srcMeta->size; i++) {
        *(dst_centers + i) = new int[3]{src_centers[i][0], src_centers[i][1], src_centers[i][2]};
    }

    dstMeta->message = (void *)dst_centers;
    dstMeta->size = srcMeta->size;

    return dstMeta;
}

/* release function set by user. "data" holds a pointer to NvDsUserMeta*/
static void object_center_free_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsCustomMsgInfo *srcMeta = (NvDsCustomMsgInfo *)user_meta->user_meta_data;
    user_meta->user_meta_data = NULL;

    if (srcMeta->message) {
        int **src_centers = (int **)srcMeta->message;
        for (size_t i = 0; i < srcMeta->size; i++) {
            g_free(src_centers[i]);
        }
        g_free(srcMeta->message);
    }
    g_free(srcMeta);
}

/**
 * Only update string label in an existing object metadata. No bounding boxes.
 * We assume only one label per object is generated
 */
static void attach_center_object(GstDsPostProcessing *dspostprocessing,
                                 NvDsObjectMeta *obj_meta,
                                 int **centers)
{
    NvDsBatchMeta *batch_meta = obj_meta->base_meta.batch_meta;

    // Attach - DsDirection MetaData
    NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    NvDsMetaType user_meta_type = NVDS_COLOR_POST_PROCESSING2_USER_OBJECT_CENTER_META;

    NvDsCustomMsgInfo *custom_msg_info = (NvDsCustomMsgInfo *)g_malloc0(sizeof(NvDsCustomMsgInfo));
    custom_msg_info->message = (void *)centers;
    custom_msg_info->size = 5; // TODO

    user_meta->user_meta_data = custom_msg_info;
    user_meta->base_meta.meta_type = user_meta_type;
    user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)object_center_copy_func;
    user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)object_center_free_func;

    nvds_add_user_meta_to_obj(obj_meta, user_meta);
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean dspostprocessing_plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_dspostprocessing_debug, "dspostprocessing", 0,
                            "dspostprocessing plugin");

    return gst_element_register(plugin, "dspostprocessing", GST_RANK_PRIMARY,
                                GST_TYPE_DSPOSTPROCESSING);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  nvdsgst_dspostprocessing,
                  DESCRIPTION,
                  dspostprocessing_plugin_init,
                  "6.0",
                  LICENSE,
                  BINARY_PACKAGE,
                  URL)
