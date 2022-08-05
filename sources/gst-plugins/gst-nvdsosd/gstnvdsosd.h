/**
 * Copyright (c) 2016-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 * version: 0.1
 */

#ifndef __GST_NVDSOSD_H__
#define __GST_NVDSOSD_H__

#include <gst/gst.h>
#include <gst/video/gstvideofilter.h>
#include <gst/video/video.h>
#include <stdlib.h>

#include "gstnvdsmeta.h"
#include "nvll_osd_api.h"

#define MAX_BG_CLR 20

G_BEGIN_DECLS
/* Standard GStreamer boilerplate */
#define GST_TYPE_NVDSOSD (gst_nvds_osd_get_type())
#define GST_NVDSOSD(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDSOSD, GstNvDsOsd))
#define GST_NVDSOSD_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVDSOSD, GstNvDsOsdClass))
#define GST_IS_NVDSOSD(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVDSOSD))
#define GST_IS_NVDSOSD_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVDSOSD))
/* Version number of package */
#define VERSION "1.8.2"
#define PACKAGE_DESCRIPTION "Gstreamer plugin to draw rectangles and text"
/* Define under which licence the package has been released */
#define PACKAGE_LICENSE "Proprietary"
#define PACKAGE_NAME "GStreamer nvosd Plugin"
/* Define to the home page for this package. */
#define PACKAGE_URL "http://nvidia.com/"
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
typedef struct _GstNvDsOsd GstNvDsOsd;
typedef struct _GstNvDsOsdClass GstNvDsOsdClass;

/**
 * GstNvDsOsd element structure.
 */
struct _GstNvDsOsd {
    /** Should be the first member when extending from GstBaseTransform. */
    GstBaseTransform parent_instance;

    /* Width of buffer. */
    gint width;
    /* Height of buffer. */
    gint height;

    /** Pointer to the nvdsosd context. */
    void *nvdsosd_context;
    /** Enum indicating how the objects are drawn,
        i.e., CPU, GPU or VIC (for Jetson only). */
    NvOSD_Mode nvdsosd_mode;

    /** Boolean value indicating whether clock is enabled. */
    gboolean show_clock;
    /** Structure containing text params for clock. */
    NvOSD_TextParams clock_text_params;

    /** List of strings to be drawn. */
    NvOSD_TextParams *text_params;
    /** List of rectangles to be drawn. */
    NvOSD_RectParams *rect_params;
    /** List of rectangles for segment masks to be drawn. */
    NvOSD_RectParams *mask_rect_params;
    /** List of segment masks to be drawn. */
    NvOSD_MaskParams *mask_params;
    /** List of land marks to be drawn. */
    NvOSD_LandmarkParams *landmark_params;
    /** List of lines to be drawn. */
    NvOSD_LineParams *line_params;
    /** List of arrows to be drawn. */
    NvOSD_ArrowParams *arrow_params;
    /** List of circles to be drawn. */
    NvOSD_CircleParams *circle_params;

    /** Number of rectangles to be drawn for a frame. */
    guint num_rect;
    /** Number of segment masks to be drawn for a frame. */
    guint num_segments;
    /** Number of strings to be drawn for a frame. */
    guint num_strings;
    /** Number of lines to be drawn for a frame. */
    guint num_lines;
    /** Number of arrows to be drawn for a frame. */
    guint num_arrows;
    /** Number of circles to be drawn for a frame. */
    guint num_circles;

    /** Structure containing details of rectangles to be drawn for a frame. */
    NvOSD_FrameRectParams *frame_rect_params;
    /** Structure containing details of segment masks to be drawn for a frame. */
    NvOSD_FrameSegmentMaskParams *frame_mask_params;
    /** Structure containing details of text to be overlayed for a frame. */
    NvOSD_FrameTextParams *frame_text_params;
    /** Structure containing details of lines to be drawn for a frame. */
    NvOSD_FrameLineParams *frame_line_params;
    /** Structure containing details of arrows to be drawn for a frame. */
    NvOSD_FrameArrowParams *frame_arrow_params;
    /** Structure containing details of circles to be drawn for a frame. */
    NvOSD_FrameCircleParams *frame_circle_params;

    /** Font of the text to be displayed. */
    gchar *font;
    /** Color of the clock, if enabled. */
    guint clock_color;
    /** Font size of the clock, if enabled. */
    guint clock_font_size;
    /** Border width of object. */
    guint border_width;
    /** Integer indicating the frame number. */
    guint frame_num;
    /** Boolean indicating whether text is to be drawn. */
    gboolean draw_text;
    /** Boolean indicating whether bounding is to be drawn. */
    gboolean draw_bbox;
    /** Boolean indicating whether instance mask is to be drawn. */
    gboolean draw_mask;
    // TODO: Landmark
    /**Array containing color info for blending */
    NvOSD_Color_info color_info[MAX_BG_CLR];
    /** Boolean indicating whether hw-blend-color-attr is set. */
    gboolean hw_blend;
    /** Integer indicating number of detected classes. */
    int num_class_entries;
    /** Integer indicating gpu id to be used. */
    guint gpu_id;
    /** Pointer to the converted buffer. */
    void *conv_buf;
};

/* GStreamer boilerplate. */
struct _GstNvDsOsdClass {
    GstBaseTransformClass parent_class;
};

GType gst_nvds_osd_get_type(void);

G_END_DECLS
#endif /* __GST_NVDSOSD_H__ */
