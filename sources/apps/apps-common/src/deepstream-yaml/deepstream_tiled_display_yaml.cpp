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

#include <iostream>
#include <string>

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"

using std::cout;
using std::endl;

gboolean parse_tiled_display_yaml(NvDsTiledDisplayConfig *config, gchar *cfg_file_path)
{
    gboolean ret = FALSE;

    YAML::Node configyml = YAML::LoadFile(cfg_file_path);
    for (YAML::const_iterator itr = configyml["tiled-display"].begin();
         itr != configyml["tiled-display"].end(); ++itr) {
        std::string paramKey = itr->first.as<std::string>();
        if (paramKey == "enable") {
            config->enable = (NvDsTiledDisplayEnable)itr->second.as<int>();
        } else if (paramKey == "rows") {
            config->rows = itr->second.as<guint>();
        } else if (paramKey == "columns") {
            config->columns = itr->second.as<guint>();
        } else if (paramKey == "width") {
            config->width = itr->second.as<guint>();
        } else if (paramKey == "height") {
            config->height = itr->second.as<guint>();
        } else if (paramKey == "gpu-id") {
            config->gpu_id = itr->second.as<guint>();
        } else if (paramKey == "nvbuf-memory-type") {
            config->nvbuf_memory_type = itr->second.as<guint>();
        } else if (paramKey == "compute-hw") {
            config->compute_hw = itr->second.as<guint>();
        } else {
            cout << "[WARNING] Unknown param found in tiled-display: " << paramKey << endl;
        }
    }
    ret = TRUE;

    if (!ret) {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}
