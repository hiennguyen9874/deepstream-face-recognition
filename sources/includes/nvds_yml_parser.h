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

/**
 * @file nvds_yml_parser.h
 * <b>NVIDIA DeepStream Yaml Parser API Specification </b>
 *
 * @b Description: This file specifies the APIs to set DeepStream GStreamer Element
 * properties by parsing YAML file.
 */

/**
 * @defgroup   yamlparser_api  DeepStream Yaml Parser API
 * Defines an API for the GStreamer NvDsYaml plugin.
 * @ingroup custom_gstreamer
 * @{
 */

#ifndef _NVGSTDS_YAML_PARSER_H_
#define _NVGSTDS_YAML_PARSER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <gst/gst.h>

/**
 * Enum for Yaml parsing status for the API call on a GstElement.
 */
typedef enum NvDsYamlParserStatus {
    /** Properties were set correctly */
    NVDS_YAML_PARSER_SUCCESS,
    /** Property group was disabled, properties were not set. */
    NVDS_YAML_PARSER_DISABLED,
    /** Encountered an error while setting properties. */
    NVDS_YAML_PARSER_ERROR
} NvDsYamlParserStatus;

/**
 * Set properties of a filesrc element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the filesrc element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_file_source(GstElement *element,
                                            gchar *cfg_file_path,
                                            const char *group);

/**
 * Set properties of a uridecodebin element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the uridecodebin element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_uridecodebin(GstElement *element,
                                             gchar *cfg_file_path,
                                             const char *group);

/**
 * Set properties of a rtspsrc element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the rtspsrc element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_rtsp_source(GstElement *element,
                                            gchar *cfg_file_path,
                                            const char *group);

/**
 * Set properties of a nvarguscamerasrc element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the nvarguscamerasrc element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_nvarguscamerasrc(GstElement *element,
                                                 gchar *cfg_file_path,
                                                 const char *group);

/**
 * Set properties of a v4l2src element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the v4l2src element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_v4l2src(GstElement *element,
                                        gchar *cfg_file_path,
                                        const char *group);

/**
 * Set properties of a multifilesrc element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the multifilesrc element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_multifilesrc(GstElement *element,
                                             gchar *cfg_file_path,
                                             const char *group);

/**
 * Set properties of a alsasrc element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the alsasrc element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_alsasrc(GstElement *element,
                                        gchar *cfg_file_path,
                                        const char *group);

/**
 * Parse semicolon separated uri(s) in the source-list group and store it in a GList
 *
 * @param[in]  src_list The empty GList address.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file on which parsing is done
 *             A key "location" is present in the group which contains semicolon
 *             separated uri(s). Once the API call finishes, the GList contains
 *             the uris(s).
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_source_list(GList **src_list,
                                            gchar *cfg_file_path,
                                            const char *group);

/**
 * Set properties of a nvstreammux element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the nvstreammux element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_streammux(GstElement *element,
                                          gchar *cfg_file_path,
                                          const char *group);

/**
 * Set properties of a nvtracker element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the nvtracker element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_tracker(GstElement *element,
                                        gchar *cfg_file_path,
                                        const char *group);

/**
 * Set properties of a nvdsosd element from values specified in a YAML configuration file.
 *
 * @param[in]  element The gst element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group The group in the YAML config file on which parsing is done
 *             and corresponding properties are set on the nvdsosd element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_osd(GstElement *element, gchar *cfg_file_path, const char *group);

/**
 * Set properties of a nvtiler element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the nvmultistreamtiler element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_tiler(GstElement *element, gchar *cfg_file_path, const char *group);

/**
 * Set properties of a nvmsgbroker element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the nmvsgbroker element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_msgbroker(GstElement *element,
                                          gchar *cfg_file_path,
                                          const char *group);

/**
 * Set properties of a nvmsgconv element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the nvmsgconv element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_msgconv(GstElement *element,
                                        gchar *cfg_file_path,
                                        const char *group);

/**
 * Set properties of a nvinfer element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the nvinfer element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_gie(GstElement *element, gchar *cfg_file_path, const char *group);

/**
 * Set properties of a nveglglessink element from values specified in a YAML configuration file.
 *
 * @param[in]  element Gstreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the nveglglessink element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_egl_sink(GstElement *element,
                                         gchar *cfg_file_path,
                                         const char *group);

/**
 * Set properties of a filesink element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the filesink element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_file_sink(GstElement *element,
                                          gchar *cfg_file_path,
                                          const char *group);

/**
 * Set properties of a fakesink element from values specified in a YAML configuration file.
 *
 * @param[in]  element GStreamer element on which properties are to be set.
 * @param[in]  cfg_file_path The YAML config file used by an application.
 * @param[in]  group Group in the YAML config file to be parsed and
 *             corresponding properties set on the fakesink element.
 * @return Yaml parsing status for the API call.
 */
NvDsYamlParserStatus nvds_parse_fake_sink(GstElement *element,
                                          gchar *cfg_file_path,
                                          const char *group);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_YAML_PARSER_H_ */

/** @} */