/**
 * Copyright (c) 2016-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 * version: 0.1
 */

#include "gstnvdsosd.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <stdio.h>

#include "nvbufsurface.h"
#include "nvtx3/nvToolsExt.h"

GST_DEBUG_CATEGORY_STATIC(gst_nvds_osd_debug);
#define GST_CAT_DEFAULT gst_nvds_osd_debug

/* For hw blending, color should be of the form:
   class_id1, R, G, B, A:class_id2, R, G, B, A */
#define DEFAULT_CLR "0,0.0,1.0,0.0,0.3:1,0.0,1.0,1.0,0.3:2,0.0,0.0,1.0,0.3:3,1.0,1.0,0.0,0.3"
#define MAX_OSD_ELEMS 1024

/* Filter signals and args */
enum {
    /* FILL ME */
    LAST_SIGNAL
};

/* Enum to identify properties */
enum {
    PROP_0,
    PROP_SHOW_CLOCK,
    PROP_SHOW_TEXT,
    PROP_CLOCK_FONT,
    PROP_CLOCK_FONT_SIZE,
    PROP_CLOCK_X_OFFSET,
    PROP_CLOCK_Y_OFFSET,
    PROP_CLOCK_COLOR,
    PROP_PROCESS_MODE,
    PROP_HW_BLEND_COLOR_ATTRS,
    PROP_GPU_DEVICE_ID,
    PROP_SHOW_BBOX,
    PROP_SHOW_MASK,
};

/* the capabilities of the inputs and outputs. */
static GstStaticPadTemplate nvdsosd_sink_factory = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM, "{ RGBA }")));

static GstStaticPadTemplate nvdsosd_src_factory = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM, "{ RGBA }")));

/* Default values for properties */
#define DEFAULT_FONT_SIZE 12
#define DEFAULT_FONT "Serif"
#ifdef PLATFORM_TEGRA
#define GST_NV_OSD_DEFAULT_PROCESS_MODE MODE_HW
#else
#define GST_NV_OSD_DEFAULT_PROCESS_MODE MODE_GPU
#endif
#define MAX_FONT_SIZE 60
#define DEFAULT_BORDER_WIDTH 4

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvds_osd_parent_class parent_class
G_DEFINE_TYPE(GstNvDsOsd, gst_nvds_osd, GST_TYPE_BASE_TRANSFORM);

#define GST_TYPE_NV_OSD_PROCESS_MODE (gst_nvds_osd_process_mode_get_type())

static GQuark _dsmeta_quark;

static GType gst_nvds_osd_process_mode_get_type(void)
{
    static GType qtype = 0;

    if (qtype == 0) {
        static const GEnumValue values[] = {
            {MODE_CPU, "CPU_MODE", "CPU_MODE"},
            {MODE_GPU, "GPU_MODE, yet to be implemented for Tegra", "GPU_MODE"},
#ifdef PLATFORM_TEGRA
            {MODE_HW, "HW_MODE. Only for Tegra. For rectdraw only.", "HW_MODE"},
#endif
            {0, NULL, NULL}};

        qtype = g_enum_register_static("GstNvDsOsdMode", values);
    }
    return qtype;
}

static void gst_nvds_osd_finalize(GObject *object);
static void gst_nvds_osd_set_property(GObject *object,
                                      guint prop_id,
                                      const GValue *value,
                                      GParamSpec *pspec);
static void gst_nvds_osd_get_property(GObject *object,
                                      guint prop_id,
                                      GValue *value,
                                      GParamSpec *pspec);
static GstFlowReturn gst_nvds_osd_transform_ip(GstBaseTransform *trans, GstBuffer *buf);
static gboolean gst_nvds_osd_start(GstBaseTransform *btrans);
static gboolean gst_nvds_osd_stop(GstBaseTransform *btrans);
static gboolean gst_nvds_osd_parse_color(GstNvDsOsd *nvdsosd, guint clock_color);

static gboolean gst_nvds_osd_parse_hw_blend_color_attrs(GstNvDsOsd *nvdsosd, const gchar *arr);
static gboolean gst_nvds_osd_get_hw_blend_color_attrs(GValue *value, GstNvDsOsd *nvdsosd);

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean gst_nvds_osd_set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
    gboolean ret = TRUE;

    GstNvDsOsd *nvdsosd = GST_NVDSOSD(trans);
    gint width = 0, height = 0;
    cudaError_t CUerr = cudaSuccess;

    nvdsosd->frame_num = 0;

    GstStructure *structure = gst_caps_get_structure(incaps, 0);

    GST_OBJECT_LOCK(nvdsosd);
    if (!gst_structure_get_int(structure, "width", &width) ||
        !gst_structure_get_int(structure, "height", &height)) {
        GST_ELEMENT_ERROR(nvdsosd, STREAM, FAILED, ("caps without width/height"), NULL);
        ret = FALSE;
        goto exit_set_caps;
    }
    if (nvdsosd->nvdsosd_context && nvdsosd->width == width && nvdsosd->height == height) {
        goto exit_set_caps;
    }

    CUerr = cudaSetDevice(nvdsosd->gpu_id);
    if (CUerr != cudaSuccess) {
        ret = FALSE;
        GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to set device"), NULL);
        goto exit_set_caps;
    }

    nvdsosd->width = width;
    nvdsosd->height = height;

    if (nvdsosd->show_clock)
        nvll_osd_set_clock_params(nvdsosd->nvdsosd_context, &nvdsosd->clock_text_params);

    nvdsosd->conv_buf =
        nvll_osd_set_params(nvdsosd->nvdsosd_context, nvdsosd->width, nvdsosd->height);

exit_set_caps:
    GST_OBJECT_UNLOCK(nvdsosd);
    return ret;
}

/**
 * Initialize all resources.
 */
static gboolean gst_nvds_osd_start(GstBaseTransform *btrans)
{
    GstNvDsOsd *nvdsosd = GST_NVDSOSD(btrans);

    cudaError_t CUerr = cudaSuccess;
    CUerr = cudaSetDevice(nvdsosd->gpu_id);
    if (CUerr != cudaSuccess) {
        GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to set device"), NULL);
        return FALSE;
    }
    GST_LOG_OBJECT(nvdsosd, "SETTING CUDA DEVICE = %d in nvdsosd func=%s\n", nvdsosd->gpu_id,
                   __func__);

    nvdsosd->nvdsosd_context = nvll_osd_create_context();

    if (nvdsosd->nvdsosd_context == NULL) {
        GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to create context nvdsosd"), NULL);
        return FALSE;
    }

    int flag_integrated = -1;
    cudaDeviceGetAttribute(&flag_integrated, cudaDevAttrIntegrated, nvdsosd->gpu_id);
    if (!flag_integrated && nvdsosd->nvdsosd_mode == MODE_HW) {
        nvdsosd->nvdsosd_mode = MODE_GPU;
    }

    if (nvdsosd->num_class_entries == 0) {
        gst_nvds_osd_parse_hw_blend_color_attrs(nvdsosd, DEFAULT_CLR);
    }

    nvll_osd_init_colors_for_hw_blend(nvdsosd->nvdsosd_context, nvdsosd->color_info,
                                      nvdsosd->num_class_entries);

    if (nvdsosd->show_clock) {
        nvll_osd_set_clock_params(nvdsosd->nvdsosd_context, &nvdsosd->clock_text_params);
    }

    return TRUE;
}

/**
 * Free up all the resources
 */
static gboolean gst_nvds_osd_stop(GstBaseTransform *btrans)
{
    GstNvDsOsd *nvdsosd = GST_NVDSOSD(btrans);

    cudaError_t CUerr = cudaSuccess;
    CUerr = cudaSetDevice(nvdsosd->gpu_id);
    if (CUerr != cudaSuccess) {
        GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to set device"), NULL);
        return FALSE;
    }
    GST_LOG_OBJECT(nvdsosd, "SETTING CUDA DEVICE = %d in nvdsosd func=%s\n", nvdsosd->gpu_id,
                   __func__);

    if (nvdsosd->nvdsosd_context)
        nvll_osd_destroy_context(nvdsosd->nvdsosd_context);

    nvdsosd->nvdsosd_context = NULL;
    nvdsosd->width = 0;
    nvdsosd->height = 0;

    return TRUE;
}

int frame_num = 0;

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn gst_nvds_osd_transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
    GstNvDsOsd *nvdsosd = GST_NVDSOSD(trans);
    GstMapInfo inmap = GST_MAP_INFO_INIT;
    unsigned int rect_cnt = 0;
    unsigned int segment_cnt = 0;
    unsigned int text_cnt = 0;
    unsigned int line_cnt = 0;
    unsigned int arrow_cnt = 0;
    unsigned int circle_cnt = 0;
    unsigned int i = 0;
    int idx = 0;
    gpointer state = NULL;
    NvBufSurface *surface = NULL;
    NvDsBatchMeta *batch_meta = NULL;

    if (!gst_buffer_map(buf, &inmap, GST_MAP_READ)) {
        GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to map info from buffer"), NULL);
        return GST_FLOW_ERROR;
    }

    nvds_set_input_system_timestamp(buf, GST_ELEMENT_NAME(nvdsosd));

    cudaError_t CUerr = cudaSuccess;
    CUerr = cudaSetDevice(nvdsosd->gpu_id);
    if (CUerr != cudaSuccess) {
        GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to set device"), NULL);
        return GST_FLOW_ERROR;
    }
    GST_LOG_OBJECT(nvdsosd, "SETTING CUDA DEVICE = %d in nvdsosd func=%s\n", nvdsosd->gpu_id,
                   __func__);

    surface = (NvBufSurface *)inmap.data;

    /* Get metadata. Update rectangle and text params */
    GstMeta *gst_meta;
    NvDsMeta *dsmeta;
    char context_name[100];
    snprintf(context_name, sizeof(context_name), "%s_(Frame=%u)", GST_ELEMENT_NAME(nvdsosd),
             nvdsosd->frame_num);
    nvtxRangePushA(context_name);
    while ((gst_meta = gst_buffer_iterate_meta(buf, &state))) {
        if (gst_meta_api_type_has_tag(gst_meta->info->api, _dsmeta_quark)) {
            dsmeta = (NvDsMeta *)gst_meta;
            if (dsmeta->meta_type == NVDS_BATCH_GST_META) {
                batch_meta = (NvDsBatchMeta *)dsmeta->meta_data;
                break;
            }
        }
    }

    NvDsMetaList *l = NULL;
    NvDsMetaList *full_obj_meta_list = NULL;
    if (batch_meta)
        full_obj_meta_list = batch_meta->obj_meta_pool->full_list;
    NvDsObjectMeta *object_meta = NULL;

    for (l = full_obj_meta_list; l != NULL; l = l->next) {
        object_meta = (NvDsObjectMeta *)(l->data);
        if (nvdsosd->draw_bbox) {
            nvdsosd->rect_params[rect_cnt] = object_meta->rect_params;
#ifdef PLATFORM_TEGRA
            /* In case of hardware blending, values set in hw-blend-color-attr
               should be considered as rect bg color values*/
            if (nvdsosd->nvdsosd_mode == MODE_HW && nvdsosd->hw_blend) {
                for (idx = 0; idx < nvdsosd->num_class_entries; idx++) {
                    if (nvdsosd->color_info[idx].id == object_meta->class_id) {
                        nvdsosd->rect_params[rect_cnt].color_id = idx;
                        nvdsosd->rect_params[rect_cnt].has_bg_color = TRUE;
                        nvdsosd->rect_params[rect_cnt].bg_color.red =
                            nvdsosd->color_info[idx].color.red;
                        nvdsosd->rect_params[rect_cnt].bg_color.blue =
                            nvdsosd->color_info[idx].color.blue;
                        nvdsosd->rect_params[rect_cnt].bg_color.green =
                            nvdsosd->color_info[idx].color.green;
                        nvdsosd->rect_params[rect_cnt].bg_color.alpha =
                            nvdsosd->color_info[idx].color.alpha;
                        break;
                    }
                }
            }
#endif
            rect_cnt++;
        }
        if (rect_cnt == MAX_OSD_ELEMS) {
            nvdsosd->frame_rect_params->num_rects = rect_cnt;
            nvdsosd->frame_rect_params->rect_params_list = nvdsosd->rect_params;
            nvdsosd->frame_rect_params->buf_ptr = &surface->surfaceList[0];
            nvdsosd->frame_rect_params->mode = nvdsosd->nvdsosd_mode;
            if (nvll_osd_draw_rectangles(nvdsosd->nvdsosd_context, nvdsosd->frame_rect_params) ==
                -1) {
                GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw rectangles"), NULL);
                return GST_FLOW_ERROR;
            }
            rect_cnt = 0;
        }
        if (nvdsosd->draw_mask && object_meta->mask_params.data &&
            object_meta->mask_params.size > 0) {
            nvdsosd->mask_rect_params[segment_cnt] = object_meta->rect_params;
            nvdsosd->mask_params[segment_cnt++] = object_meta->mask_params;
            if (segment_cnt == MAX_OSD_ELEMS) {
                nvdsosd->frame_mask_params->num_segments = segment_cnt;
                nvdsosd->frame_mask_params->rect_params_list = nvdsosd->mask_rect_params;
                nvdsosd->frame_mask_params->mask_params_list = nvdsosd->mask_params;
                nvdsosd->frame_mask_params->buf_ptr = &surface->surfaceList[0];
                nvdsosd->frame_mask_params->mode = nvdsosd->nvdsosd_mode;
                if (nvll_osd_draw_segment_masks(nvdsosd->nvdsosd_context,
                                                nvdsosd->frame_mask_params) == -1) {
                    GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw rectangles"),
                                      NULL);
                    return GST_FLOW_ERROR;
                }
                segment_cnt = 0;
            }
        }
        if (object_meta->text_params.display_text)
            nvdsosd->text_params[text_cnt++] = object_meta->text_params;
        if (text_cnt == MAX_OSD_ELEMS) {
            nvdsosd->frame_text_params->num_strings = text_cnt;
            nvdsosd->frame_text_params->text_params_list = nvdsosd->text_params;
            nvdsosd->frame_text_params->buf_ptr = &surface->surfaceList[0];
            nvdsosd->frame_text_params->mode = nvdsosd->nvdsosd_mode;
            if (nvll_osd_put_text(nvdsosd->nvdsosd_context, nvdsosd->frame_text_params) == -1) {
                GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw text"), NULL);
                return GST_FLOW_ERROR;
            }
            text_cnt = 0;
        }
    }

    NvDsMetaList *display_meta_list = NULL;
    if (batch_meta)
        display_meta_list = batch_meta->display_meta_pool->full_list;
    NvDsDisplayMeta *display_meta = NULL;

    /* Get objects to be drawn from display meta.
     * Draw objects if count equals MAX_OSD_ELEMS.
     */
    for (l = display_meta_list; l != NULL; l = l->next) {
        display_meta = (NvDsDisplayMeta *)(l->data);

        unsigned int cnt = 0;
        for (cnt = 0; cnt < display_meta->num_rects; cnt++) {
            nvdsosd->rect_params[rect_cnt++] = display_meta->rect_params[cnt];
            if (rect_cnt == MAX_OSD_ELEMS) {
                nvdsosd->frame_rect_params->num_rects = rect_cnt;
                nvdsosd->frame_rect_params->rect_params_list = nvdsosd->rect_params;
                nvdsosd->frame_rect_params->buf_ptr = &surface->surfaceList[0];
                nvdsosd->frame_rect_params->mode = nvdsosd->nvdsosd_mode;
                if (nvll_osd_draw_rectangles(nvdsosd->nvdsosd_context,
                                             nvdsosd->frame_rect_params) == -1) {
                    GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw rectangles"),
                                      NULL);
                    return GST_FLOW_ERROR;
                }
                rect_cnt = 0;
            }
        }

        for (cnt = 0; cnt < display_meta->num_labels; cnt++) {
            if (display_meta->text_params[cnt].display_text) {
                nvdsosd->text_params[text_cnt++] = display_meta->text_params[cnt];
                if (text_cnt == MAX_OSD_ELEMS) {
                    nvdsosd->frame_text_params->num_strings = text_cnt;
                    nvdsosd->frame_text_params->text_params_list = nvdsosd->text_params;
                    nvdsosd->frame_text_params->buf_ptr = &surface->surfaceList[0];
                    nvdsosd->frame_text_params->mode = nvdsosd->nvdsosd_mode;
                    if (nvll_osd_put_text(nvdsosd->nvdsosd_context, nvdsosd->frame_text_params) ==
                        -1) {
                        GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw text"), NULL);
                        return GST_FLOW_ERROR;
                    }
                    text_cnt = 0;
                }
            }
        }

        for (cnt = 0; cnt < display_meta->num_lines; cnt++) {
            nvdsosd->line_params[line_cnt++] = display_meta->line_params[cnt];
            if (line_cnt == MAX_OSD_ELEMS) {
                nvdsosd->frame_line_params->num_lines = line_cnt;
                nvdsosd->frame_line_params->line_params_list = nvdsosd->line_params;
                nvdsosd->frame_line_params->buf_ptr = &surface->surfaceList[0];
                nvdsosd->frame_line_params->mode = nvdsosd->nvdsosd_mode;
                if (nvll_osd_draw_lines(nvdsosd->nvdsosd_context, nvdsosd->frame_line_params) ==
                    -1) {
                    GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw lines"), NULL);
                    return GST_FLOW_ERROR;
                }
                line_cnt = 0;
            }
        }

        for (cnt = 0; cnt < display_meta->num_arrows; cnt++) {
            nvdsosd->arrow_params[arrow_cnt++] = display_meta->arrow_params[cnt];
            if (arrow_cnt == MAX_OSD_ELEMS) {
                nvdsosd->frame_arrow_params->num_arrows = arrow_cnt;
                nvdsosd->frame_arrow_params->arrow_params_list = nvdsosd->arrow_params;
                nvdsosd->frame_arrow_params->buf_ptr = &surface->surfaceList[0];
                nvdsosd->frame_arrow_params->mode = nvdsosd->nvdsosd_mode;
                if (nvll_osd_draw_arrows(nvdsosd->nvdsosd_context, nvdsosd->frame_arrow_params) ==
                    -1) {
                    GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw arrows"), NULL);
                    return GST_FLOW_ERROR;
                }
                arrow_cnt = 0;
            }
        }

        for (cnt = 0; cnt < display_meta->num_circles; cnt++) {
            nvdsosd->circle_params[circle_cnt++] = display_meta->circle_params[cnt];
            if (circle_cnt == MAX_OSD_ELEMS) {
                nvdsosd->frame_circle_params->num_circles = circle_cnt;
                nvdsosd->frame_circle_params->circle_params_list = nvdsosd->circle_params;
                nvdsosd->frame_circle_params->buf_ptr = &surface->surfaceList[0];
                nvdsosd->frame_circle_params->mode = nvdsosd->nvdsosd_mode;
                if (nvll_osd_draw_circles(nvdsosd->nvdsosd_context, nvdsosd->frame_circle_params) ==
                    -1) {
                    GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw circles"), NULL);
                    return GST_FLOW_ERROR;
                }
                circle_cnt = 0;
            }
        }
        i++;
    }

    nvdsosd->num_rect = rect_cnt;
    nvdsosd->num_segments = segment_cnt;
    nvdsosd->num_strings = text_cnt;
    nvdsosd->num_lines = line_cnt;
    nvdsosd->num_arrows = arrow_cnt;
    nvdsosd->num_circles = circle_cnt;
    if (rect_cnt != 0 && nvdsosd->draw_bbox) {
        nvdsosd->frame_rect_params->num_rects = nvdsosd->num_rect;
        nvdsosd->frame_rect_params->rect_params_list = nvdsosd->rect_params;
        nvdsosd->frame_rect_params->buf_ptr = &surface->surfaceList[0];
        nvdsosd->frame_rect_params->mode = nvdsosd->nvdsosd_mode;
        if (nvll_osd_draw_rectangles(nvdsosd->nvdsosd_context, nvdsosd->frame_rect_params) == -1) {
            GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw rectangles"), NULL);
            return GST_FLOW_ERROR;
        }
    }

    if (segment_cnt != 0 && nvdsosd->draw_mask) {
        nvdsosd->frame_mask_params->num_segments = nvdsosd->num_segments;
        nvdsosd->frame_mask_params->rect_params_list = nvdsosd->mask_rect_params;
        nvdsosd->frame_mask_params->mask_params_list = nvdsosd->mask_params;
        nvdsosd->frame_mask_params->buf_ptr = &surface->surfaceList[0];
        nvdsosd->frame_mask_params->mode = nvdsosd->nvdsosd_mode;
        if (nvll_osd_draw_segment_masks(nvdsosd->nvdsosd_context, nvdsosd->frame_mask_params) ==
            -1) {
            GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw segment masks"), NULL);
            return GST_FLOW_ERROR;
        }
    }

    if ((nvdsosd->show_clock || text_cnt) && nvdsosd->draw_text) {
        nvdsosd->frame_text_params->num_strings = nvdsosd->num_strings;
        nvdsosd->frame_text_params->text_params_list = nvdsosd->text_params;
        nvdsosd->frame_text_params->buf_ptr = &surface->surfaceList[0];
        nvdsosd->frame_text_params->mode = nvdsosd->nvdsosd_mode;
        if (nvll_osd_put_text(nvdsosd->nvdsosd_context, nvdsosd->frame_text_params) == -1) {
            GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw text"), NULL);
            return GST_FLOW_ERROR;
        }
    }

    if (line_cnt != 0) {
        nvdsosd->frame_line_params->num_lines = nvdsosd->num_lines;
        nvdsosd->frame_line_params->line_params_list = nvdsosd->line_params;
        nvdsosd->frame_line_params->buf_ptr = &surface->surfaceList[0];
        nvdsosd->frame_line_params->mode = nvdsosd->nvdsosd_mode;
        if (nvll_osd_draw_lines(nvdsosd->nvdsosd_context, nvdsosd->frame_line_params) == -1) {
            GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw lines"), NULL);
            return GST_FLOW_ERROR;
        }
    }

    if (arrow_cnt != 0) {
        nvdsosd->frame_arrow_params->num_arrows = nvdsosd->num_arrows;
        nvdsosd->frame_arrow_params->arrow_params_list = nvdsosd->arrow_params;
        nvdsosd->frame_arrow_params->buf_ptr = &surface->surfaceList[0];
        nvdsosd->frame_arrow_params->mode = nvdsosd->nvdsosd_mode;
        if (nvll_osd_draw_arrows(nvdsosd->nvdsosd_context, nvdsosd->frame_arrow_params) == -1) {
            GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw arrows"), NULL);
            return GST_FLOW_ERROR;
        }
    }

    if (circle_cnt != 0) {
        nvdsosd->frame_circle_params->num_circles = nvdsosd->num_circles;
        nvdsosd->frame_circle_params->circle_params_list = nvdsosd->circle_params;
        nvdsosd->frame_circle_params->buf_ptr = &surface->surfaceList[0];
        nvdsosd->frame_circle_params->mode = nvdsosd->nvdsosd_mode;
        if (nvll_osd_draw_circles(nvdsosd->nvdsosd_context, nvdsosd->frame_circle_params) == -1) {
            GST_ELEMENT_ERROR(nvdsosd, RESOURCE, FAILED, ("Unable to draw circles"), NULL);
            return GST_FLOW_ERROR;
        }
    }

    nvtxRangePop();
    nvdsosd->frame_num++;

    nvds_set_output_system_timestamp(buf, GST_ELEMENT_NAME(nvdsosd));

    gst_buffer_unmap(buf, &inmap);
    return GST_FLOW_OK;
}

/* Called when the plugin is destroyed.
 * Free all structures which have been malloc'd.
 */
static void gst_nvds_osd_finalize(GObject *object)
{
    GstNvDsOsd *nvdsosd = GST_NVDSOSD(object);

    if (nvdsosd->clock_text_params.font_params.font_name) {
        g_free((char *)nvdsosd->clock_text_params.font_params.font_name);
    }
    g_free(nvdsosd->rect_params);
    g_free(nvdsosd->mask_rect_params);
    g_free(nvdsosd->mask_params);
    g_free(nvdsosd->landmark_params);
    g_free(nvdsosd->text_params);
    g_free(nvdsosd->line_params);
    g_free(nvdsosd->arrow_params);
    g_free(nvdsosd->circle_params);

    g_free(nvdsosd->frame_rect_params);
    g_free(nvdsosd->frame_mask_params);
    g_free(nvdsosd->frame_text_params);
    g_free(nvdsosd->frame_line_params);
    g_free(nvdsosd->frame_arrow_params);
    g_free(nvdsosd->frame_circle_params);

    G_OBJECT_CLASS(parent_class)->finalize(object);
}

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void gst_nvds_osd_class_init(GstNvDsOsdClass *klass)
{
    GObjectClass *gobject_class;
    GstElementClass *gstelement_class;
    GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class = (GObjectClass *)klass;
    gstelement_class = (GstElementClass *)klass;

    base_transform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_nvds_osd_transform_ip);
    base_transform_class->start = GST_DEBUG_FUNCPTR(gst_nvds_osd_start);
    base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_nvds_osd_stop);
    base_transform_class->set_caps = GST_DEBUG_FUNCPTR(gst_nvds_osd_set_caps);

    gobject_class->set_property = gst_nvds_osd_set_property;
    gobject_class->get_property = gst_nvds_osd_get_property;
    gobject_class->finalize = gst_nvds_osd_finalize;

    base_transform_class->passthrough_on_same_caps = TRUE;

    g_object_class_install_property(
        gobject_class, PROP_SHOW_CLOCK,
        g_param_spec_boolean("display-clock", "clock", "Whether to display clock", FALSE,
                             G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_SHOW_TEXT,
        g_param_spec_boolean("display-text", "text", "Whether to display text", TRUE,
                             G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_SHOW_BBOX,
        g_param_spec_boolean("display-bbox", "text", "Whether to display bounding boxes", TRUE,
                             G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_SHOW_MASK,
        g_param_spec_boolean("display-mask", "text", "Whether to display instance mask", TRUE,
                             G_PARAM_READWRITE));

    // TODO: Landmark

    g_object_class_install_property(
        gobject_class, PROP_CLOCK_FONT,
        g_param_spec_string("clock-font", "clock-font", "Clock Font to be set", "DEFAULT_FONT",
                            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_CLOCK_FONT_SIZE,
        g_param_spec_uint("clock-font-size", "clock-font-size", "font size of the clock", 0,
                          MAX_FONT_SIZE, DEFAULT_FONT_SIZE,
                          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY));

    g_object_class_install_property(
        gobject_class, PROP_CLOCK_X_OFFSET,
        g_param_spec_uint("x-clock-offset", "x-clock-offset", "x-clock-offset", 0, G_MAXUINT, 0,
                          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY));

    g_object_class_install_property(
        gobject_class, PROP_CLOCK_Y_OFFSET,
        g_param_spec_uint("y-clock-offset", "y-clock-offset", "y-clock-offset", 0, G_MAXUINT, 0,
                          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY));

    g_object_class_install_property(
        gobject_class, PROP_CLOCK_COLOR,
        g_param_spec_uint("clock-color", "clock-color", "clock-color", 0, G_MAXUINT, G_MAXUINT,
                          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY));

    g_object_class_install_property(
        gobject_class, PROP_PROCESS_MODE,
        g_param_spec_enum("process-mode", "Process Mode", "Rect and text draw process mode",
                          GST_TYPE_NV_OSD_PROCESS_MODE, GST_NV_OSD_DEFAULT_PROCESS_MODE,
                          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY));

    g_object_class_install_property(
        gobject_class, PROP_HW_BLEND_COLOR_ATTRS,
        g_param_spec_string("hw-blend-color-attr", "HW Blend Color Attr",
                            "color attributes for all classes,\n"
                            "\t\t\t Use string with values of color class atrributes \n"
                            "\t\t\t in ClassID (int), r(float), g(float), b(float), a(float)\n"
                            "\t\t\t in order to set the property.\n"
                            "\t\t\t Applicable only for HW mode on Jetson.\n"
                            "\t\t\t e.g. 0,0.0,1.0,0.0,0.3:1,1.0,0.0,0.3,0.3",
                            DEFAULT_CLR,
                            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_GPU_DEVICE_ID,
        g_param_spec_uint("gpu-id", "Set GPU Device ID", "Set GPU Device ID", 0, G_MAXUINT, 0,
                          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY));

    gst_element_class_set_details_simple(
        gstelement_class, "NvDsOsd plugin", "NvDsOsd functionality",
        "Gstreamer bounding box draw element",
        "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
        "@ https://devtalk.nvidia.com/default/board/209/");

    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&nvdsosd_src_factory));
    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&nvdsosd_sink_factory));

    _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void gst_nvds_osd_set_property(GObject *object,
                                      guint prop_id,
                                      const GValue *value,
                                      GParamSpec *pspec)
{
    GstNvDsOsd *nvdsosd = GST_NVDSOSD(object);

    switch (prop_id) {
    case PROP_SHOW_CLOCK:
        nvdsosd->show_clock = g_value_get_boolean(value);
        break;
    case PROP_SHOW_TEXT:
        nvdsosd->draw_text = g_value_get_boolean(value);
        break;
    case PROP_SHOW_BBOX:
        nvdsosd->draw_bbox = g_value_get_boolean(value);
        break;
    case PROP_SHOW_MASK:
        nvdsosd->draw_mask = g_value_get_boolean(value);
        break;
    case PROP_CLOCK_FONT:
        if (nvdsosd->clock_text_params.font_params.font_name) {
            g_free((char *)nvdsosd->clock_text_params.font_params.font_name);
        }
        nvdsosd->clock_text_params.font_params.font_name = (gchar *)g_value_dup_string(value);
        break;
    case PROP_CLOCK_FONT_SIZE:
        nvdsosd->clock_text_params.font_params.font_size = g_value_get_uint(value);
        break;
    case PROP_CLOCK_X_OFFSET:
        nvdsosd->clock_text_params.x_offset = g_value_get_uint(value);
        break;
    case PROP_CLOCK_Y_OFFSET:
        nvdsosd->clock_text_params.y_offset = g_value_get_uint(value);
        break;
    case PROP_CLOCK_COLOR:
        gst_nvds_osd_parse_color(nvdsosd, g_value_get_uint(value));
        break;
    case PROP_PROCESS_MODE:
        nvdsosd->nvdsosd_mode = (NvOSD_Mode)g_value_get_enum(value);
        break;
    case PROP_HW_BLEND_COLOR_ATTRS:
        nvdsosd->hw_blend = TRUE;
        gst_nvds_osd_parse_hw_blend_color_attrs(nvdsosd, g_value_get_string(value));
        break;
    case PROP_GPU_DEVICE_ID:
        nvdsosd->gpu_id = g_value_get_uint(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void gst_nvds_osd_get_property(GObject *object,
                                      guint prop_id,
                                      GValue *value,
                                      GParamSpec *pspec)
{
    GstNvDsOsd *nvdsosd = GST_NVDSOSD(object);

    switch (prop_id) {
    case PROP_SHOW_CLOCK:
        g_value_set_boolean(value, nvdsosd->show_clock);
        break;
    case PROP_SHOW_TEXT:
        g_value_set_boolean(value, nvdsosd->draw_text);
        break;
    case PROP_SHOW_BBOX:
        g_value_set_boolean(value, nvdsosd->draw_bbox);
        break;
    case PROP_SHOW_MASK:
        g_value_set_boolean(value, nvdsosd->draw_mask);
        break;
    case PROP_CLOCK_FONT:
        g_value_set_string(value, nvdsosd->font);
        break;
    case PROP_CLOCK_FONT_SIZE:
        g_value_set_uint(value, nvdsosd->clock_font_size);
        break;
    case PROP_CLOCK_X_OFFSET:
        g_value_set_uint(value, nvdsosd->clock_text_params.x_offset);
        break;
    case PROP_CLOCK_Y_OFFSET:
        g_value_set_uint(value, nvdsosd->clock_text_params.y_offset);
        break;
    case PROP_CLOCK_COLOR:
        g_value_set_uint(value, nvdsosd->clock_color);
        break;
    case PROP_PROCESS_MODE:
        g_value_set_enum(value, nvdsosd->nvdsosd_mode);
        break;
    case PROP_HW_BLEND_COLOR_ATTRS:
        gst_nvds_osd_get_hw_blend_color_attrs(value, nvdsosd);
        break;
    case PROP_GPU_DEVICE_ID:
        g_value_set_uint(value, nvdsosd->gpu_id);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/* Set default values of certain properties.
 */
static void gst_nvds_osd_init(GstNvDsOsd *nvdsosd)
{
    nvdsosd->show_clock = FALSE;
    nvdsosd->draw_text = TRUE;
    nvdsosd->draw_bbox = TRUE;
    nvdsosd->draw_mask = FALSE;
    nvdsosd->clock_text_params.font_params.font_name = g_strdup(DEFAULT_FONT);
    nvdsosd->clock_text_params.font_params.font_size = DEFAULT_FONT_SIZE;
    nvdsosd->nvdsosd_mode = GST_NV_OSD_DEFAULT_PROCESS_MODE;
    nvdsosd->border_width = DEFAULT_BORDER_WIDTH;
    nvdsosd->num_rect = 0;
    nvdsosd->num_segments = 0;
    nvdsosd->num_strings = 0;
    nvdsosd->num_lines = 0;
    nvdsosd->clock_text_params.font_params.font_color.red = 1.0;
    nvdsosd->clock_text_params.font_params.font_color.green = 0.0;
    nvdsosd->clock_text_params.font_params.font_color.blue = 0.0;
    nvdsosd->clock_text_params.font_params.font_color.alpha = 1.0;
    nvdsosd->rect_params = g_new0(NvOSD_RectParams, MAX_OSD_ELEMS);
    nvdsosd->mask_rect_params = g_new0(NvOSD_RectParams, MAX_OSD_ELEMS);
    nvdsosd->mask_params = g_new0(NvOSD_MaskParams, MAX_OSD_ELEMS);
    nvdsosd->landmark_params = g_new0(NvOSD_LandmarkParams, MAX_OSD_ELEMS);
    nvdsosd->text_params = g_new0(NvOSD_TextParams, MAX_OSD_ELEMS);
    nvdsosd->line_params = g_new0(NvOSD_LineParams, MAX_OSD_ELEMS);
    nvdsosd->arrow_params = g_new0(NvOSD_ArrowParams, MAX_OSD_ELEMS);
    nvdsosd->circle_params = g_new0(NvOSD_CircleParams, MAX_OSD_ELEMS);
    nvdsosd->frame_rect_params = g_new0(NvOSD_FrameRectParams, MAX_OSD_ELEMS);
    nvdsosd->frame_mask_params = g_new0(NvOSD_FrameSegmentMaskParams, MAX_OSD_ELEMS);
    nvdsosd->frame_text_params = g_new0(NvOSD_FrameTextParams, MAX_OSD_ELEMS);
    nvdsosd->frame_line_params = g_new0(NvOSD_FrameLineParams, MAX_OSD_ELEMS);
    nvdsosd->frame_arrow_params = g_new0(NvOSD_FrameArrowParams, MAX_OSD_ELEMS);
    nvdsosd->frame_circle_params = g_new0(NvOSD_FrameCircleParams, MAX_OSD_ELEMS);
    nvdsosd->hw_blend = FALSE;
}

/**
 * Set color of text for clock, if enabled.
 */
static gboolean gst_nvds_osd_parse_color(GstNvDsOsd *nvdsosd, guint clock_color)
{
    nvdsosd->clock_text_params.font_params.font_color.red =
        (gfloat)((clock_color & 0xff000000) >> 24) / 255;
    nvdsosd->clock_text_params.font_params.font_color.green =
        (gfloat)((clock_color & 0x00ff0000) >> 16) / 255;
    nvdsosd->clock_text_params.font_params.font_color.blue =
        (gfloat)((clock_color & 0x0000ff00) >> 8) / 255;
    nvdsosd->clock_text_params.font_params.font_color.alpha =
        (gfloat)((clock_color & 0x000000ff)) / 255;
    return TRUE;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean nvdsosd_init(GstPlugin *nvdsosd)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvds_osd_debug, "nvdsosd", 0, "nvdsosd plugin");

    return gst_element_register(nvdsosd, "nvdsosd", GST_RANK_PRIMARY, GST_TYPE_NVDSOSD);
}

static gboolean gst_nvds_osd_parse_hw_blend_color_attrs(GstNvDsOsd *nvdsosd, const gchar *arr)
{
    gchar *str = (gchar *)arr;
    int idx = 0;
    int class_id = 0;

    while (str != NULL && str[0] != '\0') {
        class_id = atoi(str);
        if (class_id >= MAX_BG_CLR) {
            g_print("nvdsosd: class_id %d is exceeding than %d\n", class_id, MAX_BG_CLR);
            exit(-1);
        }
        nvdsosd->color_info[idx].id = class_id;
        str = g_strstr_len(str, -1, ",") + 1;

        nvdsosd->color_info[idx].color.red = atof(str);
        str = g_strstr_len(str, -1, ",") + 1;
        nvdsosd->color_info[idx].color.green = atof(str);
        str = g_strstr_len(str, -1, ",") + 1;
        nvdsosd->color_info[idx].color.blue = atof(str);
        str = g_strstr_len(str, -1, ",") + 1;
        nvdsosd->color_info[idx].color.alpha = atof(str);
        str = g_strstr_len(str, -1, ":");

        if (str) {
            str = str + 1;
        }
        idx++;
        if (idx >= MAX_BG_CLR) {
            g_print("idx (%d) entries exceeded MAX_CLASSES %d\n", idx, MAX_BG_CLR);
            break;
        }
    }

    nvdsosd->num_class_entries = idx;
    return TRUE;
}

static gboolean gst_nvds_osd_get_hw_blend_color_attrs(GValue *value, GstNvDsOsd *nvdsosd)
{
    int idx = 0;
    gchar arr[100];

    while (idx < (nvdsosd->num_class_entries - 1)) {
        sprintf(arr, "%d,%f,%f,%f,%f:", nvdsosd->color_info[idx].id,
                nvdsosd->color_info[idx].color.red, nvdsosd->color_info[idx].color.green,
                nvdsosd->color_info[idx].color.blue, nvdsosd->color_info[idx].color.alpha);
        idx++;
    }
    sprintf(arr, "%d,%f,%f,%f,%f:", nvdsosd->color_info[idx].id, nvdsosd->color_info[idx].color.red,
            nvdsosd->color_info[idx].color.green, nvdsosd->color_info[idx].color.blue,
            nvdsosd->color_info[idx].color.alpha);

    g_value_set_string(value, arr);
    return TRUE;
}

#ifndef PACKAGE
#define PACKAGE "nvdsosd"
#endif

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  nvdsgst_osd,
                  PACKAGE_DESCRIPTION,
                  nvdsosd_init,
                  "6.0",
                  PACKAGE_LICENSE,
                  PACKAGE_NAME,
                  PACKAGE_URL)
