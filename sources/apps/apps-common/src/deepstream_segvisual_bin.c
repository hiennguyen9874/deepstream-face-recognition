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

#include "deepstream_common.h"
#include "deepstream_segvisual.h"

gboolean create_segvisual_bin(NvSegVisualConfig *config, NvSegVisualBin *bin)
{
    gboolean ret = FALSE;

    bin->bin = gst_bin_new("segvisual_bin");
    if (!bin->bin) {
        NVGSTDS_ERR_MSG_V("Failed to create 'segvisual_bin'");
        goto done;
    }

    bin->queue = gst_element_factory_make(NVDS_ELEM_QUEUE, "segvisual_queue");
    if (!bin->queue) {
        NVGSTDS_ERR_MSG_V("Failed to create 'segvisual_queue'");
        goto done;
    }

    bin->nvsegvisual = gst_element_factory_make(NVDS_ELEM_SEGVISUAL, "nvsegvisual0");
    if (!bin->nvsegvisual) {
        NVGSTDS_ERR_MSG_V("Failed to create 'nvsegvisual0'");
        goto done;
    }

    gst_bin_add_many(GST_BIN(bin->bin), bin->queue, bin->nvsegvisual, NULL);

    g_object_set(G_OBJECT(bin->nvsegvisual), "batch-size", config->batch_size, NULL);
    g_object_set(G_OBJECT(bin->nvsegvisual), "width", config->width, NULL);
    g_object_set(G_OBJECT(bin->nvsegvisual), "height", config->height, NULL);

    NVGSTDS_LINK_ELEMENT(bin->queue, bin->nvsegvisual);

    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->queue, "sink");

    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->nvsegvisual, "src");

    ret = TRUE;
done:
    if (!ret) {
        NVGSTDS_ERR_MSG_V("%s failed", __func__);
    }
    return ret;
}
