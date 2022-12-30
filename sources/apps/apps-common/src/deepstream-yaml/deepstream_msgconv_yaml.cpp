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

gboolean parse_msgconv_yaml(NvDsSinkMsgConvBrokerConfig *config,
                            std::string group_str,
                            gchar *cfg_file_path)
{
    gboolean ret = FALSE;
    YAML::Node configyml = YAML::LoadFile(cfg_file_path);
    char *group = (char *)malloc(sizeof(char) * 1024);
    std::strncpy(group, group_str.c_str(), 1024);

    for (YAML::const_iterator itr = configyml[group_str].begin(); itr != configyml[group_str].end();
         ++itr) {
        std::string paramKey = itr->first.as<std::string>();

        if (paramKey == "enable") {
            config->enable = itr->second.as<gboolean>();
        } else if (paramKey == "msg-conv-config") {
            std::string temp = itr->second.as<std::string>();
            char *str = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(str, temp.c_str(), 1024);
            config->config_file_path = (char *)malloc(sizeof(char) * 1024);
            if (!get_absolute_file_path_yaml(cfg_file_path, str, config->config_file_path)) {
                g_printerr("Error: Could not parse msg-conv-config in %s.\n", group);
                g_free(str);
                goto done;
            }
            g_free(str);
        } else if (paramKey == "msg-conv-payload-type") {
            config->conv_payload_type = itr->second.as<guint>();
        } else if (paramKey == "msg-conv-msg2p-lib") {
            std::string temp = itr->second.as<std::string>();
            char *str = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(str, temp.c_str(), 1024);
            config->conv_msg2p_lib = (char *)malloc(sizeof(char) * 1024);
            if (!get_absolute_file_path_yaml(cfg_file_path, str, config->conv_msg2p_lib)) {
                g_printerr("Error: Could not parse msg-conv-msg2p-lib in %s.\n", group);
                g_free(str);
                goto done;
            }
            g_free(str);
        } else if (paramKey == "msg-conv-comp-id") {
            config->conv_comp_id = itr->second.as<guint>();
        } else if (paramKey == "debug-payload-dir") {
            std::string temp = itr->second.as<std::string>();
            char *str = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(str, temp.c_str(), 1024);
            config->debug_payload_dir = (char *)malloc(sizeof(char) * 1024);
            if (!get_absolute_file_path_yaml(cfg_file_path, str, config->debug_payload_dir)) {
                g_printerr("Error: Could not parse debug-payload-dir in %s.\n", group);
                g_free(str);
                goto done;
            }
            g_free(str);
        } else if (paramKey == "multiple-payloads") {
            config->multiple_payloads = itr->second.as<gboolean>();
        } else if (paramKey == "msg-conv-msg2p-new-api") {
            config->conv_msg2p_new_api = itr->second.as<gboolean>();
        } else if (paramKey == "msg-conv-frame-interval") {
            config->conv_frame_interval = itr->second.as<guint>();
        }
    }

    ret = TRUE;
done:
    if (!ret) {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}