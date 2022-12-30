/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <cstring>
#include <iostream>
#include <string>

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"

using std::cout;
using std::endl;

gboolean parse_streammux_yaml(NvDsStreammuxConfig *config, gchar *cfg_file_path)
{
    gboolean ret = FALSE;

    config->batched_push_timeout = -1;
    config->attach_sys_ts_as_ntp = TRUE;

    YAML::Node configyml = YAML::LoadFile(cfg_file_path);
    for (YAML::const_iterator itr = configyml["streammux"].begin();
         itr != configyml["streammux"].end(); ++itr) {
        std::string paramKey = itr->first.as<std::string>();
        if (paramKey == "width") {
            config->pipeline_width = itr->second.as<gint>();
        } else if (paramKey == "height") {
            config->pipeline_height = itr->second.as<gint>();
        } else if (paramKey == "gpu-id") {
            config->gpu_id = itr->second.as<guint>();
        } else if (paramKey == "live-source") {
            config->live_source = itr->second.as<gboolean>();
        } else if (paramKey == "buffer-pool-size") {
            config->buffer_pool_size = itr->second.as<gint>();
        } else if (paramKey == "batch-size") {
            config->batch_size = itr->second.as<gint>();
        } else if (paramKey == "batched-push-timeout") {
            config->batched_push_timeout = itr->second.as<gint>();
        } else if (paramKey == "enable-padding") {
            config->enable_padding = itr->second.as<gboolean>();
        } else if (paramKey == "nvbuf-memory-type") {
            config->nvbuf_memory_type = itr->second.as<guint>();
        } else if (paramKey == "config-file") {
            std::string temp = itr->second.as<std::string>();
            char *str = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(str, temp.c_str(), 1024);
            config->config_file_path = (char *)malloc(sizeof(char) * 1024);
            if (!get_absolute_file_path_yaml(cfg_file_path, str, config->config_file_path)) {
                g_printerr("Error: Could not parse config-file in streammux\n");
                g_free(str);
                goto done;
            }
        } else if (paramKey == "compute-hw") {
            config->compute_hw = itr->second.as<gint>();
        } else if (paramKey == "attach-sys-ts") {
            config->attach_sys_ts_as_ntp = itr->second.as<gboolean>();
        } else if (paramKey == "frame-num-reset-on-stream-reset") {
            config->frame_num_reset_on_stream_reset = itr->second.as<gboolean>();
        } else if (paramKey == "frame-num-reset-on-eos") {
            config->frame_num_reset_on_eos = itr->second.as<gboolean>();
        } else if (paramKey == "num-surfaces-per-frame") {
            config->num_surface_per_frame = itr->second.as<gint>();
        } else if (paramKey == "interpolation-method") {
            config->interpolation_method = itr->second.as<gint>();
        } else if (paramKey == "sync-inputs") {
            config->sync_inputs = itr->second.as<gboolean>();
        } else if (paramKey == "max-latency") {
            config->max_latency = itr->second.as<guint64>();
        } else {
            cout << "[WARNING] Unknown param found in streammux: " << paramKey << endl;
            goto done;
        }
    }
    config->is_parsed = TRUE;

    ret = TRUE;
done:
    if (!ret) {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}
