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

#include "deepstream_dspostprocessing.h"

#include "deepstream_common.h"

// Create bin, add queue and the element, link all elements and ghost pads,
// Set the element properties from the parsed config
gboolean create_dspostprocessing_bin(NvDsDsPostProcessingConfig *config,
                                     NvDsDsPostProcessingBin *bin)
{
    GstCaps *caps = NULL;
    gboolean ret = FALSE;

    bin->bin = gst_bin_new("dspostprocessing_bin");
    if (!bin->bin) {
        NVGSTDS_ERR_MSG_V("Failed to create 'dspostprocessing_bin'");
        goto done;
    }

    bin->queue = gst_element_factory_make(NVDS_ELEM_QUEUE, "dspostprocessing_queue");
    if (!bin->queue) {
        NVGSTDS_ERR_MSG_V("Failed to create 'dspostprocessing_queue'");
        goto done;
    }

    bin->elem_dspostprocessing =
        gst_element_factory_make(NVDS_ELEM_DSPOSTPROCESSING_ELEMENT, "dspostprocessing0");
    if (!bin->elem_dspostprocessing) {
        NVGSTDS_ERR_MSG_V("Failed to create 'dspostprocessing0'");
        goto done;
    }

    bin->pre_conv = gst_element_factory_make(NVDS_ELEM_VIDEO_CONV, "dspostprocessing_conv0");
    if (!bin->pre_conv) {
        NVGSTDS_ERR_MSG_V("Failed to create 'dspostprocessing_conv0'");
        goto done;
    }

    bin->cap_filter = gst_element_factory_make(NVDS_ELEM_CAPS_FILTER, "dspostprocessing_caps");
    if (!bin->cap_filter) {
        NVGSTDS_ERR_MSG_V("Failed to create 'dspostprocessing_caps'");
        goto done;
    }

    caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGBA", NULL);

    GstCapsFeatures *feature = NULL;
    feature = gst_caps_features_new(MEMORY_FEATURES, NULL);
    gst_caps_set_features(caps, 0, feature);

    g_object_set(G_OBJECT(bin->cap_filter), "caps", caps, NULL);

    gst_caps_unref(caps);

    gst_bin_add_many(GST_BIN(bin->bin), bin->queue, bin->pre_conv, bin->cap_filter,
                     bin->elem_dspostprocessing, NULL);

    NVGSTDS_LINK_ELEMENT(bin->queue, bin->pre_conv);
    NVGSTDS_LINK_ELEMENT(bin->pre_conv, bin->cap_filter);
    NVGSTDS_LINK_ELEMENT(bin->cap_filter, bin->elem_dspostprocessing);

    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->queue, "sink");

    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->elem_dspostprocessing, "src");

    g_object_set(G_OBJECT(bin->elem_dspostprocessing), "unique-id", config->unique_id, "gpu-id",
                 config->gpu_id, NULL);

    if (config->num_initial_colors > 0) {
        gchar initial_colors_str[512];
        initial_colors_str[0] = '\0';
        for (gint i = 0; i < config->num_initial_colors; i++) {
            g_snprintf(initial_colors_str + strlen(initial_colors_str),
                       sizeof(initial_colors_str) - strlen(initial_colors_str) - 1, "%d;",
                       config->list_initial_colors[i]);
        }
        g_object_set(G_OBJECT(bin->elem_dspostprocessing), "initial-colors", initial_colors_str,
                     NULL);
    }

    g_object_set(G_OBJECT(bin->pre_conv), "gpu-id", config->gpu_id, NULL);

    g_object_set(G_OBJECT(bin->pre_conv), "nvbuf-memory-type", config->nvbuf_memory_type, NULL);

    ret = TRUE;

done:
    if (!ret) {
        NVGSTDS_ERR_MSG_V("%s failed", __func__);
    }

    return ret;
}
