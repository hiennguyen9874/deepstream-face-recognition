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

#ifndef _NVGSTDS_STREAMMUX_YAML_H_
#define _NVGSTDS_STREAMMUX_YAML_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#include <yaml-cpp/yaml.h>
#pragma GCC diagnostic pop

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "deepstream_c2d_msg.h"
#include "deepstream_dewarper.h"
#include "deepstream_dsanalytics.h"
#include "deepstream_dsexample.h"
#include "deepstream_gie.h"
#include "deepstream_image_save.h"
#include "deepstream_osd.h"
#include "deepstream_preprocess.h"
#include "deepstream_sinks.h"
#include "deepstream_sources.h"
#include "deepstream_streammux.h"
#include "deepstream_tiled_display.h"
#include "deepstream_tracker.h"

#define _MAX_STR_LENGTH 1024

std::vector<std::string> split_string(std::string input);

gboolean get_absolute_file_path_yaml(const gchar *cfg_file_path,
                                     const gchar *file_path,
                                     char *abs_path_str);

gboolean parse_streammux_yaml(NvDsStreammuxConfig *config, gchar *cfg_file_path);

gboolean parse_tiled_display_yaml(NvDsTiledDisplayConfig *config, gchar *cfg_file_path);

gboolean parse_osd_yaml(NvDsOSDConfig *config, gchar *cfg_file_path);

gboolean parse_image_save_yaml(NvDsImageSave *config, gchar *cfg_file_path);

gboolean parse_msgconsumer_yaml(NvDsMsgConsumerConfig *config,
                                std::string group,
                                gchar *cfg_file_path);

gboolean parse_msgconv_yaml(NvDsSinkMsgConvBrokerConfig *config,
                            std::string group,
                            gchar *cfg_file_path);

gboolean parse_sink_yaml(NvDsSinkSubBinConfig *config, std::string group, gchar *cfg_file_path);

gboolean parse_source_yaml(NvDsSourceConfig *config,
                           std::vector<std::string> headers,
                           std::vector<std::string> source_values,
                           gchar *cfg_file_path);

gboolean parse_tracker_yaml(NvDsTrackerConfig *config, gchar *cfg_file_path);

gboolean parse_gie_yaml(NvDsGieConfig *config, std::string group, gchar *cfg_file_path);

gboolean parse_preprocess_yaml(NvDsPreProcessConfig *config, gchar *cfg_file_path);

gboolean parse_dewarper_yaml(NvDsDewarperConfig *config, gchar *cfg_file_path);

gboolean parse_dsexample_yaml(NvDsDsExampleConfig *config, gchar *cfg_file_path);

gboolean parse_dsanalytics_yaml(NvDsDsAnalyticsConfig *config, gchar *cfg_file_path);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_DSEXAMPLE_H_ */
