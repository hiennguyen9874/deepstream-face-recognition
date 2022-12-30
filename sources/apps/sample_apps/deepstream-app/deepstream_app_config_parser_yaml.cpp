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

#include <stdlib.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "deepstream_app.h"
#include "deepstream_config_yaml.h"

using std::cout;
using std::endl;

static gboolean parse_tests_yaml(NvDsConfig *config, gchar *cfg_file_path)
{
    gboolean ret = FALSE;
    YAML::Node configyml = YAML::LoadFile(cfg_file_path);

    for (YAML::const_iterator itr = configyml["tests"].begin(); itr != configyml["tests"].end();
         ++itr) {
        std::string paramKey = itr->first.as<std::string>();
        if (paramKey == "file-loop") {
            config->file_loop = itr->second.as<gint>();
        } else {
            cout << "Unknown key " << paramKey << " for group tests" << endl;
        }
    }

    ret = TRUE;

    if (!ret) {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}

static gboolean parse_app_yaml(NvDsConfig *config, gchar *cfg_file_path)
{
    gboolean ret = FALSE;
    YAML::Node configyml = YAML::LoadFile(cfg_file_path);

    for (YAML::const_iterator itr = configyml["application"].begin();
         itr != configyml["application"].end(); ++itr) {
        std::string paramKey = itr->first.as<std::string>();
        if (paramKey == "enable-perf-measurement") {
            config->enable_perf_measurement = itr->second.as<gboolean>();
        } else if (paramKey == "perf-measurement-interval-sec") {
            config->perf_measurement_interval_sec = itr->second.as<guint>();
        } else if (paramKey == "gie-kitti-output-dir") {
            std::string temp = itr->second.as<std::string>();
            char *str = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(str, temp.c_str(), 1024);
            config->bbox_dir_path = (char *)malloc(sizeof(char) * 1024);
            get_absolute_file_path_yaml(cfg_file_path, str, config->bbox_dir_path);
            g_free(str);
        } else if (paramKey == "kitti-track-output-dir") {
            std::string temp = itr->second.as<std::string>();
            char *str = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(str, temp.c_str(), 1024);
            config->kitti_track_dir_path = (char *)malloc(sizeof(char) * 1024);
            get_absolute_file_path_yaml(cfg_file_path, str, config->kitti_track_dir_path);
            g_free(str);
        } else {
            cout << "Unknown key " << paramKey << " for group application" << endl;
        }
    }

    ret = TRUE;

    if (!ret) {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}

static std::vector<std::string> split_csv_entries(std::string input)
{
    std::vector<int> positions;
    for (unsigned int i = 0; i < input.size(); i++) {
        if (input[i] == ',')
            positions.push_back(i);
    }
    std::vector<std::string> ret;
    int prev = 0;
    for (auto &j : positions) {
        std::string temp = input.substr(prev, j - prev);
        ret.push_back(temp);
        prev = j + 1;
    }
    ret.push_back(input.substr(prev, input.size() - prev));
    return ret;
}

gboolean parse_config_file_yaml(NvDsConfig *config, gchar *cfg_file_path)
{
    gboolean parse_err = false;
    gboolean ret = FALSE;
    YAML::Node configyml = YAML::LoadFile(cfg_file_path);
    std::string source_str = "source";
    std::string sink_str = "sink";
    std::string sgie_str = "secondary-gie";
    std::string msgcons_str = "message-consumer";

    config->source_list_enabled = FALSE;

    for (YAML::const_iterator itr = configyml.begin(); itr != configyml.end(); ++itr) {
        std::string paramKey = itr->first.as<std::string>();

        if (paramKey == "application") {
            parse_err = !parse_app_yaml(config, cfg_file_path);
        } else if (paramKey == "source") {
            if (configyml["source"]["csv-file-path"]) {
                std::string csv_file_path = configyml["source"]["csv-file-path"].as<std::string>();
                char *str = (char *)malloc(sizeof(char) * 1024);
                std::strncpy(str, csv_file_path.c_str(), 1024);
                char *abs_csv_path = (char *)malloc(sizeof(char) * 1024);
                get_absolute_file_path_yaml(cfg_file_path, str, abs_csv_path);
                g_free(str);

                std::ifstream inputFile(abs_csv_path);
                if (!inputFile.is_open()) {
                    cout << "Couldn't open CSV file " << abs_csv_path << endl;
                }
                std::string line, temp;
                /* Separating header field and inserting as strings into the vector.
                 */
                getline(inputFile, line);
                std::vector<std::string> headers = split_csv_entries(line);
                /*Parsing each csv entry as an input source */
                while (getline(inputFile, line)) {
                    std::vector<std::string> source_values = split_csv_entries(line);
                    if (config->num_source_sub_bins == MAX_SOURCE_BINS) {
                        NVGSTDS_ERR_MSG_V("App supports max %d sources", MAX_SOURCE_BINS);
                        ret = FALSE;
                        goto done;
                    }
                    guint source_id = 0;
                    source_id = config->num_source_sub_bins;
                    parse_err = !parse_source_yaml(&config->multi_source_config[source_id], headers,
                                                   source_values, cfg_file_path);
                    if (config->multi_source_config[source_id].enable)
                        config->num_source_sub_bins++;
                }
            } else {
                NVGSTDS_ERR_MSG_V("CSV file not specified\n");
                ret = FALSE;
                goto done;
            }
        } else if (paramKey == "streammux") {
            parse_err = !parse_streammux_yaml(&config->streammux_config, cfg_file_path);
        } else if (paramKey == "osd") {
            parse_err = !parse_osd_yaml(&config->osd_config, cfg_file_path);
        } else if (paramKey == "pre-process") {
            parse_err = !parse_preprocess_yaml(&config->preprocess_config, cfg_file_path);
        } else if (paramKey == "primary-gie") {
            parse_err = !parse_gie_yaml(&config->primary_gie_config, paramKey, cfg_file_path);
        } else if (paramKey == "tracker") {
            parse_err = !parse_tracker_yaml(&config->tracker_config, cfg_file_path);
        } else if (paramKey.compare(0, sgie_str.size(), sgie_str) == 0) {
            if (config->num_secondary_gie_sub_bins == MAX_SECONDARY_GIE_BINS) {
                NVGSTDS_ERR_MSG_V("App supports max %d secondary GIEs", MAX_SECONDARY_GIE_BINS);
                ret = FALSE;
                goto done;
            }
            parse_err = !parse_gie_yaml(
                &config->secondary_gie_sub_bin_config[config->num_secondary_gie_sub_bins], paramKey,
                cfg_file_path);
            if (config->secondary_gie_sub_bin_config[config->num_secondary_gie_sub_bins].enable) {
                config->num_secondary_gie_sub_bins++;
            }
        } else if (paramKey.compare(0, sink_str.size(), sink_str) == 0) {
            if (config->num_sink_sub_bins == MAX_SINK_BINS) {
                NVGSTDS_ERR_MSG_V("App supports max %d sinks", MAX_SINK_BINS);
                ret = FALSE;
                goto done;
            }
            parse_err =
                !parse_sink_yaml(&config->sink_bin_sub_bin_config[config->num_sink_sub_bins],
                                 paramKey, cfg_file_path);
            if (config->sink_bin_sub_bin_config[config->num_sink_sub_bins].enable) {
                config->num_sink_sub_bins++;
            }
        } else if (paramKey.compare(0, msgcons_str.size(), msgcons_str) == 0) {
            if (config->num_message_consumers == MAX_MESSAGE_CONSUMERS) {
                NVGSTDS_ERR_MSG_V("App supports max %d consumers", MAX_MESSAGE_CONSUMERS);
                ret = FALSE;
                goto done;
            }
            parse_err = !parse_msgconsumer_yaml(
                &config->message_consumer_config[config->num_message_consumers], paramKey,
                cfg_file_path);

            if (config->message_consumer_config[config->num_message_consumers].enable) {
                config->num_message_consumers++;
            }
        } else if (paramKey == "tiled-display") {
            parse_err = !parse_tiled_display_yaml(&config->tiled_display_config, cfg_file_path);
        } else if (paramKey == "img-save") {
            parse_err = !parse_image_save_yaml(&config->image_save_config, cfg_file_path);
        } else if (paramKey == "nvds-analytics") {
            parse_err = !parse_dsanalytics_yaml(&config->dsanalytics_config, cfg_file_path);
        } else if (paramKey == "ds-example") {
            parse_err = !parse_dsexample_yaml(&config->dsexample_config, cfg_file_path);
        } else if (paramKey == "message-converter") {
            parse_err = !parse_msgconv_yaml(&config->msg_conv_config, paramKey, cfg_file_path);
        } else if (paramKey == "tests") {
            parse_err = !parse_tests_yaml(config, cfg_file_path);
        }

        if (parse_err) {
            cout << "failed parsing" << endl;
            goto done;
        }
    }
    /* Updating batch size when source list is enabled */
    /* if (config->source_list_enabled == TRUE) {
        // For streammux and pgie, batch size is set to number of sources
        config->streammux_config.batch_size = config->num_source_sub_bins;
        config->primary_gie_config.batch_size = config->num_source_sub_bins;
        if (config->sgie_batch_size != 0) {
            for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
                config->secondary_gie_sub_bin_config[i].batch_size = config->sgie_batch_size;
            }
        }
    } */
    unsigned int i, j;
    for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
        if (config->secondary_gie_sub_bin_config[i].unique_id ==
            config->primary_gie_config.unique_id) {
            NVGSTDS_ERR_MSG_V("Non unique gie ids found");
            ret = FALSE;
            goto done;
        }
    }

    for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
        for (j = i + 1; j < config->num_secondary_gie_sub_bins; j++) {
            if (config->secondary_gie_sub_bin_config[i].unique_id ==
                config->secondary_gie_sub_bin_config[j].unique_id) {
                NVGSTDS_ERR_MSG_V("Non unique gie id %d found",
                                  config->secondary_gie_sub_bin_config[i].unique_id);
                ret = FALSE;
                goto done;
            }
        }
    }

    for (i = 0; i < config->num_source_sub_bins; i++) {
        if (config->multi_source_config[i].type == NV_DS_SOURCE_URI_MULTIPLE) {
            if (config->multi_source_config[i].num_sources < 1) {
                config->multi_source_config[i].num_sources = 1;
            }
            for (j = 1; j < config->multi_source_config[i].num_sources; j++) {
                if (config->num_source_sub_bins == MAX_SOURCE_BINS) {
                    NVGSTDS_ERR_MSG_V("App supports max %d sources", MAX_SOURCE_BINS);
                    ret = FALSE;
                    goto done;
                }
                memcpy(&config->multi_source_config[config->num_source_sub_bins],
                       &config->multi_source_config[i], sizeof(config->multi_source_config[i]));
                config->multi_source_config[config->num_source_sub_bins].type = NV_DS_SOURCE_URI;
                config->multi_source_config[config->num_source_sub_bins].uri = g_strdup_printf(
                    config->multi_source_config[config->num_source_sub_bins].uri, j);
                config->num_source_sub_bins++;
            }
            config->multi_source_config[i].type = NV_DS_SOURCE_URI;
            config->multi_source_config[i].uri =
                g_strdup_printf(config->multi_source_config[i].uri, 0);
        }
    }

    ret = TRUE;
done:
    if (!ret) {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}
