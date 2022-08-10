/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

/////////////////
/* Start Custom */
/////////////////
#include <cuda_runtime_api.h>
#include <glib.h>
#include <gst/gst.h>
////////////////
/* End Custom */
////////////////

#include <json-glib/json-glib.h>

/////////////////
/* Start Custom */
/////////////////
#include <math.h>

#include <limits>
////////////////
/* End Custom */
////////////////

#include <stdlib.h>
#include <uuid.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include "deepstream_schema.h"

/////////////////
/* Start Custom */
/////////////////
#include "gstnvdsinfer.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvdsmeta_schema.h"
////////////////
/* End Custom */
////////////////

using namespace std;

#define MAX_TIME_STAMP_LEN (64)

/////////////////
/* Start Custom */
/////////////////

#define NVDS_COLOR_POST_PROCESSING_USER_FRAME_META \
    nvds_get_user_meta_type(((gchar *)"NVIDIA.COLOR.POST.PROCESSING.USER.FRAME.META"))

#define NVDS_COLOR_POST_PROCESSING_USER_OBJECT_META \
    nvds_get_user_meta_type(((gchar *)"NVIDIA.COLOR.POST.PROCESSING.USER.OBJECT.META"))

#define NVDS_IMG_CROP_OBJECT_USER_OBJECT_META                               \
    nvds_get_user_meta_type(((gchar *)"NVIDIA.IMG.CROP.OBJECT.USER.OBJECT." \
                                      "META"))

#define NVDS_MASK_CROP_OBJECT_USER_OBJECT_META                               \
    nvds_get_user_meta_type(((gchar *)"NVIDIA.MASK.CROP.OBJECT.USER.OBJECT." \
                                      "META"))

#define NVDS_COLOR_POST_PROCESSING2_USER_OBJECT_CENTER_META \
    nvds_get_user_meta_type(((gchar *)"NVIDIA.COLOR.POST.PROCESSING2.USER.OBJECT.CENTER.META"))

#define NVDS_COLOR_POST_PROCESSING2_USER_OBJECT_COUTER_META \
    nvds_get_user_meta_type(((gchar *)"NVIDIA.COLOR.POST.PROCESSING2.USER.OBJECT.COUTER.META"))

////////////////
/* End Custom */
////////////////

static void generate_ts_rfc3339(char *buf, int buf_size)
{
    time_t tloc;
    struct tm tm_log;
    struct timespec ts;
    char strmsec[6]; //.nnnZ\0

    clock_gettime(CLOCK_REALTIME, &ts);
    memcpy(&tloc, (void *)(&ts.tv_sec), sizeof(time_t));
    gmtime_r(&tloc, &tm_log);
    strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
    int ms = ts.tv_nsec / 1000000;
    g_snprintf(strmsec, sizeof(strmsec), ".%.3dZ", ms);
    strncat(buf, strmsec, buf_size);
}

/////////////////
/* Start Custom */
/////////////////
static GstClockTime generate_ts_rfc3339_from_ts(char *buf, int buf_size, GstClockTime ts)
{
    time_t tloc;
    struct tm tm_log;
    char strmsec[6];
    int ms;
    GstClockTime ts_generated;
    /** ts itself is UTC Time in ns */
    struct timespec timespec_current;
    GST_TIME_TO_TIMESPEC(ts, timespec_current);
    memcpy(&tloc, (void *)(&timespec_current.tv_sec), sizeof(time_t));
    ms = timespec_current.tv_nsec / 1000000;
    ts_generated = ts;

    gmtime_r(&tloc, &tm_log);
    strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
    g_snprintf(strmsec, sizeof(strmsec), ".%.3dZ", ms);
    strncat(buf, strmsec, buf_size);
    return ts_generated;
}
////////////////
/* End Custom */
////////////////

static JsonObject *generate_place_object(void *privData, NvDsFrameMeta *frame_meta)
{
    NvDsPayloadPriv *privObj = NULL;
    NvDsPlaceObject *dsPlaceObj = NULL;
    JsonObject *placeObj;
    JsonObject *jobject;
    JsonObject *jobject2;

    privObj = (NvDsPayloadPriv *)privData;
    auto idMap = privObj->placeObj.find(frame_meta->source_id);

    if (idMap != privObj->placeObj.end()) {
        dsPlaceObj = &idMap->second;
    } else {
        cout << "No entry for " CONFIG_GROUP_PLACE << frame_meta->source_id
             << " in configuration file" << endl;
        return NULL;
    }

    /* place object
   * "place":
     {
       "id": "string",
       "name": "endeavor",
       “type”: “garage”,
       "location": {
         "lat": 30.333,
         "lon": -40.555,
         "alt": 100.00
       },
       "entrance/aisle": {
         "name": "walsh",
         "lane": "lane1",
         "level": "P2",
         "coordinate": {
           "x": 1.0,
           "y": 2.0,
           "z": 3.0
         }
       }
     }
   */

    placeObj = json_object_new();
    json_object_set_string_member(placeObj, "id", dsPlaceObj->id.c_str());
    json_object_set_string_member(placeObj, "name", dsPlaceObj->name.c_str());
    json_object_set_string_member(placeObj, "type", dsPlaceObj->type.c_str());

    // location sub object
    jobject = json_object_new();
    json_object_set_double_member(jobject, "lat", dsPlaceObj->location[0]);
    json_object_set_double_member(jobject, "lon", dsPlaceObj->location[1]);
    json_object_set_double_member(jobject, "alt", dsPlaceObj->location[2]);
    json_object_set_object_member(placeObj, "location", jobject);

    // place sub object (user to provide the name for sub place ex: parkingSpot/aisle/entrance..etc
    jobject = json_object_new();

    json_object_set_string_member(jobject, "id", dsPlaceObj->subObj.field1.c_str());
    json_object_set_string_member(jobject, "name", dsPlaceObj->subObj.field2.c_str());
    json_object_set_string_member(jobject, "level", dsPlaceObj->subObj.field3.c_str());
    json_object_set_object_member(placeObj, "place-sub-field", jobject);

    // coordinates for place sub object
    jobject2 = json_object_new();
    json_object_set_double_member(jobject2, "x", dsPlaceObj->coordinate[0]);
    json_object_set_double_member(jobject2, "y", dsPlaceObj->coordinate[1]);
    json_object_set_double_member(jobject2, "z", dsPlaceObj->coordinate[2]);
    json_object_set_object_member(jobject, "coordinate", jobject2);

    return placeObj;
}

static JsonObject *generate_sensor_object(void *privData, NvDsFrameMeta *frame_meta)
{
    NvDsPayloadPriv *privObj = NULL;
    NvDsSensorObject *dsSensorObj = NULL;
    JsonObject *sensorObj;
    JsonObject *jobject;

    privObj = (NvDsPayloadPriv *)privData;
    auto idMap = privObj->sensorObj.find(frame_meta->source_id);

    if (idMap != privObj->sensorObj.end()) {
        dsSensorObj = &idMap->second;
    } else {
        cout << "No entry for " CONFIG_GROUP_SENSOR << frame_meta->source_id
             << " in configuration file" << endl;
        return NULL;
    }

    /* sensor object
   * "sensor": {
       "id": "string",
       "type": "Camera/Puck",
       "location": {
         "lat": 45.99,
         "lon": 35.54,
         "alt": 79.03
       },
       "coordinate": {
         "x": 5.2,
         "y": 10.1,
         "z": 11.2
       },
       "description": "Entrance of Endeavor Garage Right Lane"
     }
   */

    // sensor object
    sensorObj = json_object_new();
    json_object_set_string_member(sensorObj, "id", dsSensorObj->id.c_str());
    json_object_set_string_member(sensorObj, "type", dsSensorObj->type.c_str());
    json_object_set_string_member(sensorObj, "description", dsSensorObj->desc.c_str());

    // location sub object
    jobject = json_object_new();
    json_object_set_double_member(jobject, "lat", dsSensorObj->location[0]);
    json_object_set_double_member(jobject, "lon", dsSensorObj->location[1]);
    json_object_set_double_member(jobject, "alt", dsSensorObj->location[2]);
    json_object_set_object_member(sensorObj, "location", jobject);

    // coordinate sub object
    jobject = json_object_new();
    json_object_set_double_member(jobject, "x", dsSensorObj->coordinate[0]);
    json_object_set_double_member(jobject, "y", dsSensorObj->coordinate[1]);
    json_object_set_double_member(jobject, "z", dsSensorObj->coordinate[2]);
    json_object_set_object_member(sensorObj, "coordinate", jobject);

    return sensorObj;
}

static JsonObject *generate_analytics_module_object(void *privData, NvDsFrameMeta *frame_meta)
{
    NvDsPayloadPriv *privObj = NULL;
    NvDsAnalyticsObject *dsObj = NULL;
    JsonObject *analyticsObj;

    privObj = (NvDsPayloadPriv *)privData;

    auto idMap = privObj->analyticsObj.find(frame_meta->source_id);

    if (idMap != privObj->analyticsObj.end()) {
        dsObj = &idMap->second;
    } else {
        cout << "No entry for " CONFIG_GROUP_ANALYTICS << frame_meta->source_id
             << " in configuration file" << endl;
        return NULL;
    }

    /* analytics object
   * "analyticsModule": {
       "id": "string",
       "description": "Vehicle Detection and License Plate Recognition",
       "confidence": 97.79,
       "source": "OpenALR",
       "version": "string"
     }
   */

    // analytics object
    analyticsObj = json_object_new();
    json_object_set_string_member(analyticsObj, "id", dsObj->id.c_str());
    json_object_set_string_member(analyticsObj, "description", dsObj->desc.c_str());
    json_object_set_string_member(analyticsObj, "source", dsObj->source.c_str());
    json_object_set_string_member(analyticsObj, "version", dsObj->version.c_str());

    return analyticsObj;
}

static JsonObject *generate_object_object(void *privData,
                                          NvDsFrameMeta *frame_meta,
                                          NvDsObjectMeta *obj_meta)
{
    JsonObject *objectObj;
    JsonObject *jobject;
    gchar tracking_id[64];
    // GList *objectMask = NULL;

    // object object
    objectObj = json_object_new();
    if (snprintf(tracking_id, sizeof(tracking_id), "%lu", obj_meta->object_id) >=
        (int)sizeof(tracking_id))
        g_warning("Not enough space to copy trackingId");

    json_object_set_string_member(objectObj, "id", tracking_id);
    json_object_set_double_member(objectObj, "speed", 0);
    json_object_set_double_member(objectObj, "direction", 0);
    json_object_set_double_member(objectObj, "orientation", 0);

    jobject = json_object_new();
    json_object_set_double_member(jobject, "confidence", obj_meta->confidence);

    // Fetch object classifiers detected
    for (NvDsClassifierMetaList *cl = obj_meta->classifier_meta_list; cl; cl = cl->next) {
        NvDsClassifierMeta *cl_meta = (NvDsClassifierMeta *)cl->data;

        for (NvDsLabelInfoList *ll = cl_meta->label_info_list; ll; ll = ll->next) {
            NvDsLabelInfo *ll_meta = (NvDsLabelInfo *)ll->data;
            if (cl_meta->classifier_type != NULL && strcmp("", cl_meta->classifier_type))
                json_object_set_string_member(jobject, cl_meta->classifier_type,
                                              ll_meta->result_label);
        }
    }
    json_object_set_object_member(objectObj, obj_meta->obj_label, jobject);

    // bbox sub object
    float scaleW = (float)frame_meta->source_frame_width / (frame_meta->pipeline_width == 0)
                       ? 1
                       : frame_meta->pipeline_width;
    float scaleH = (float)frame_meta->source_frame_height / (frame_meta->pipeline_height == 0)
                       ? 1
                       : frame_meta->pipeline_height;

    float left = obj_meta->rect_params.left * scaleW;
    float top = obj_meta->rect_params.top * scaleH;
    float width = obj_meta->rect_params.width * scaleW;
    float height = obj_meta->rect_params.height * scaleH;

    jobject = json_object_new();
    json_object_set_int_member(jobject, "topleftx", left);
    json_object_set_int_member(jobject, "toplefty", top);
    json_object_set_int_member(jobject, "bottomrightx", left + width);
    json_object_set_int_member(jobject, "bottomrighty", top + height);
    json_object_set_object_member(objectObj, "bbox", jobject);

    // location sub object
    jobject = json_object_new();
    json_object_set_object_member(objectObj, "location", jobject);

    // coordinate sub object
    jobject = json_object_new();
    json_object_set_object_member(objectObj, "coordinate", jobject);

    return objectObj;
}

static JsonObject *generate_event_object(NvDsObjectMeta *obj_meta)
{
    JsonObject *eventObj;
    uuid_t uuid;
    gchar uuidStr[37];

    /*
   * "event": {
       "id": "event-id",
       "type": "entry / exit"
     }
   */

    uuid_generate_random(uuid);
    uuid_unparse_lower(uuid, uuidStr);

    eventObj = json_object_new();
    json_object_set_string_member(eventObj, "id", uuidStr);
    json_object_set_string_member(eventObj, "type", "");
    return eventObj;
}

/////////////////
/* Start Custom */
/////////////////

static JsonObject *generate_object_object_custom(void *privData,
                                                 NvDsFrameMeta *frame_meta,
                                                 NvDsObjectMeta *obj_meta)
{
    JsonObject *objectObj;
    JsonObject *jobject;
    JsonObject *jobjectAttribute;
    JsonObject *jobjectAttributeElement;
    gchar tracking_id[64];
    // gchar parent_id[64];
    // GList *objectMask = NULL;

    // object object
    objectObj = json_object_new();

    if (obj_meta->object_id != UNTRACKED_OBJECT_ID) {
        if (snprintf(tracking_id, sizeof(tracking_id), "%lu", obj_meta->object_id) >=
            (int)sizeof(tracking_id))
            g_warning("Not enough space to copy trackingId");
        json_object_set_string_member(objectObj, "id", tracking_id);
    } else
        json_object_set_string_member(objectObj, "id", "");

    // if (obj_meta->unique_component_id == 2) {
    //     if (obj_meta->parent) {
    //         if (snprintf(parent_id, sizeof(parent_id), "%lu", obj_meta->parent->object_id) >=
    //             (int)sizeof(parent_id))
    //             g_warning("Not enough space to copy parentId");
    //         json_object_set_string_member(objectObj, "parentId", parent_id);
    //     } else
    //         json_object_set_string_member(objectObj, "parentId", "");
    // } else
    //     json_object_set_string_member(objectObj, "parentId", "");

    // json_object_set_double_member(objectObj, "speed", 0);
    // json_object_set_double_member(objectObj, "direction", 0);
    // json_object_set_double_member(objectObj, "orientation", 0);

    json_object_set_string_member(objectObj, "label", obj_meta->obj_label);
    json_object_set_double_member(objectObj, "confidence", obj_meta->confidence);

    for (NvDsClassifierMetaList *cl = obj_meta->classifier_meta_list; cl; cl = cl->next) {
        NvDsClassifierMeta *cl_meta = (NvDsClassifierMeta *)cl->data;

        if (cl_meta->unique_component_id == 2) {
            jobjectAttribute = json_object_new();

            for (NvDsLabelInfoList *ll = cl_meta->label_info_list; ll; ll = ll->next) {
                NvDsLabelInfo *ll_meta = (NvDsLabelInfo *)ll->data;

                jobjectAttributeElement = json_object_new();
                json_object_set_int_member(jobjectAttributeElement, "index", ll_meta->label_id);
                json_object_set_double_member(jobjectAttributeElement, "confidence",
                                              ll_meta->result_prob);

                json_object_set_object_member(jobjectAttribute, ll_meta->result_label,
                                              jobjectAttributeElement);
            }
            json_object_set_object_member(objectObj, "attribute", jobjectAttribute);
        } else {
            // TODO: Warnings not implemented
            // std::cout << cl_meta->classifier_type << " "
            //           << strcmp("attributerecognition", cl_meta->classifier_type) << std::endl;
        }
    }

    for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL;
         l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;

        if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDSINFER_TENSOR_OUTPUT_META) {
            /* convert to tensor metadata */
            NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;

            for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                info->buffer = meta->out_buf_ptrs_host[i];
            }

            NvDsInferDimsCHW dims;
            getDimsCHWFromDims(dims, meta->output_layers_info[0].inferDims);
            unsigned int featureDim = dims.c;

            float *outputCoverageBuffer = (float *)meta->output_layers_info[0].buffer;

            JsonArray *feature_vector = json_array_new();
            for (unsigned int c = 0; c < featureDim; c++) {
                if (outputCoverageBuffer[c] == std::numeric_limits<double>::infinity()) {
                    // TODO: Wanrnings this is error
                    json_array_add_double_element(feature_vector, 999999);
                } else if (outputCoverageBuffer[c] == -std::numeric_limits<double>::infinity()) {
                    json_array_add_double_element(feature_vector, -999999);
                } else {
                    json_array_add_double_element(feature_vector, outputCoverageBuffer[c]);
                }
            }
            json_object_set_array_member(objectObj, "featureVector", feature_vector);
        }

        if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDSINFER_LANDMARK_META) {
            NvDSInferLandmarkMeta *landmark_meta =
                (NvDSInferLandmarkMeta *)user_meta->user_meta_data;

            JsonArray *landmark = json_array_new();

            for (unsigned int landmark_id = 0; landmark_id < landmark_meta->num_landmark;
                 landmark_id++) {
                json_array_add_double_element(landmark, landmark_meta->data[2 * landmark_id]);
                json_array_add_double_element(landmark, landmark_meta->data[2 * landmark_id + 1]);
            }
            json_object_set_array_member(objectObj, "landmark", landmark);
        }

        if (user_meta->base_meta.meta_type == NVDS_IMG_CROP_OBJECT_USER_OBJECT_META) {
            // FilePath from ds-example
            json_object_set_string_member(objectObj, "filePath",
                                          (gchar *)user_meta->user_meta_data);
        }

        if (user_meta->base_meta.meta_type == NVDS_CUSTOM_MSG_BLOB) {
            NvDsCustomMsgInfo *custom_blob = (NvDsCustomMsgInfo *)user_meta->user_meta_data;
            string msg = string((const char *)custom_blob->message, custom_blob->size);

            GError *err = NULL;
            JsonNode *jnode = json_from_string(msg.c_str(), &err);

            JsonObject *jdata = json_node_get_object(jnode);
            GList *keys = json_object_get_members(jdata);

            for (GList *k = keys; k; k = g_list_next(k)) {
                JsonNode *node_value = json_object_get_member(jdata, (gchar *)k->data);
                if (json_object_has_member(objectObj, (gchar *)k->data)) {
                    // TODO: Warnings
                }
                json_object_set_member(objectObj, (gchar *)k->data, node_value);
            }
        }
    }

    float left = obj_meta->rect_params.left / ((frame_meta->pipeline_width == 0)
                                                   ? (float)frame_meta->source_frame_width
                                                   : frame_meta->pipeline_width);
    float top = obj_meta->rect_params.top / ((frame_meta->pipeline_height == 0)
                                                 ? (float)frame_meta->source_frame_height
                                                 : frame_meta->pipeline_height);
    float width = obj_meta->rect_params.width / ((frame_meta->pipeline_width == 0)
                                                     ? (float)frame_meta->source_frame_width
                                                     : frame_meta->pipeline_width);
    float height = obj_meta->rect_params.height / ((frame_meta->pipeline_height == 0)
                                                       ? (float)frame_meta->source_frame_height
                                                       : frame_meta->pipeline_height);

    jobject = json_object_new();
    json_object_set_double_member(jobject, "topleftx", left);
    json_object_set_double_member(jobject, "toplefty", top);
    json_object_set_double_member(jobject, "bottomrightx", left + width);
    json_object_set_double_member(jobject, "bottomrighty", top + height);
    json_object_set_object_member(objectObj, "bbox", jobject);

    // // location sub object
    // jobject = json_object_new();
    // json_object_set_object_member(objectObj, "location", jobject);

    // // coordinate sub object
    // jobject = json_object_new();
    // json_object_set_object_member(objectObj, "coordinate", jobject);

    return objectObj;
}

gchar *generate_dsmeta_message_custom(void *privData, void *frameMeta, void *objMeta)
{
    JsonNode *rootNode;
    JsonObject *rootObj;
    // JsonObject *placeObj;
    // JsonObject *sensorObj;
    // JsonObject *analyticsObj;
    // JsonObject *eventObj;
    JsonObject *objectObj;
    gchar *message;

    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)frameMeta;
    NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)objMeta;

    uuid_t msgId;
    gchar msgIdStr[37];

    uuid_generate_random(msgId);
    uuid_unparse_lower(msgId, msgIdStr);

    // // place object
    // placeObj = generate_place_object(privData, frame_meta);

    // // sensor object
    // sensorObj = generate_sensor_object(privData, frame_meta);

    // // analytics object
    // analyticsObj = generate_analytics_module_object(privData, frame_meta);

    // object object
    objectObj = generate_object_object_custom(privData, frame_meta, obj_meta);

    // // event object
    // eventObj = generate_event_object(obj_meta);

    char ts[MAX_TIME_STAMP_LEN + 1];
    generate_ts_rfc3339(ts, MAX_TIME_STAMP_LEN);

    // root object
    rootObj = json_object_new();
    json_object_set_string_member(rootObj, "messageid", msgIdStr);
    // json_object_set_string_member(rootObj, "mdsversion", "1.0");
    json_object_set_string_member(rootObj, "@timestamp", ts);
    // json_object_set_object_member(rootObj, "place", placeObj);
    // json_object_set_object_member(rootObj, "sensor", sensorObj);
    // json_object_set_object_member(rootObj, "analyticsModule", analyticsObj);
    json_object_set_object_member(rootObj, "object", objectObj);
    // json_object_set_object_member(rootObj, "event", eventObj);

    // json_object_set_string_member(rootObj, "videoPath", "");

    // Search for any custom message blob within frame usermeta list
    JsonArray *jArray = json_array_new();
    for (NvDsUserMetaList *l = frame_meta->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *frame_usermeta = (NvDsUserMeta *)l->data;
        if (frame_usermeta && frame_usermeta->base_meta.meta_type == NVDS_CUSTOM_MSG_BLOB) {
            NvDsCustomMsgInfo *custom_blob = (NvDsCustomMsgInfo *)frame_usermeta->user_meta_data;
            string msg = string((const char *)custom_blob->message, custom_blob->size);
            json_array_add_string_element(jArray, msg.c_str());
        }
    }
    if (json_array_get_length(jArray) > 0)
        json_object_set_array_member(rootObj, "customMessage", jArray);
    else
        json_array_unref(jArray);

    rootNode = json_node_new(JSON_NODE_OBJECT);
    json_node_set_object(rootNode, rootObj);

    message = json_to_string(rootNode, TRUE);
    json_node_free(rootNode);
    json_object_unref(rootObj);

    return message;
}

////////////////
/* End Custom */
////////////////

gchar *generate_dsmeta_message(void *privData, void *frameMeta, void *objMeta)
{
    JsonNode *rootNode;
    JsonObject *rootObj;
    JsonObject *placeObj;
    JsonObject *sensorObj;
    JsonObject *analyticsObj;
    JsonObject *eventObj;
    JsonObject *objectObj;
    gchar *message;

    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)frameMeta;
    NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)objMeta;

    uuid_t msgId;
    gchar msgIdStr[37];

    uuid_generate_random(msgId);
    uuid_unparse_lower(msgId, msgIdStr);

    // place object
    placeObj = generate_place_object(privData, frame_meta);

    // sensor object
    sensorObj = generate_sensor_object(privData, frame_meta);

    // analytics object
    analyticsObj = generate_analytics_module_object(privData, frame_meta);

    // object object
    objectObj = generate_object_object(privData, frame_meta, obj_meta);
    // event object
    eventObj = generate_event_object(obj_meta);

    char ts[MAX_TIME_STAMP_LEN + 1];
    generate_ts_rfc3339(ts, MAX_TIME_STAMP_LEN);

    // root object
    rootObj = json_object_new();
    json_object_set_string_member(rootObj, "messageid", msgIdStr);
    json_object_set_string_member(rootObj, "mdsversion", "1.0");
    json_object_set_string_member(rootObj, "@timestamp", ts);
    json_object_set_object_member(rootObj, "place", placeObj);
    json_object_set_object_member(rootObj, "sensor", sensorObj);
    json_object_set_object_member(rootObj, "analyticsModule", analyticsObj);
    json_object_set_object_member(rootObj, "object", objectObj);
    json_object_set_object_member(rootObj, "event", eventObj);

    json_object_set_string_member(rootObj, "videoPath", "");

    // Search for any custom message blob within frame usermeta list
    JsonArray *jArray = json_array_new();

    for (NvDsUserMetaList *l = frame_meta->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *frame_usermeta = (NvDsUserMeta *)l->data;
        if (frame_usermeta && frame_usermeta->base_meta.meta_type == NVDS_CUSTOM_MSG_BLOB) {
            NvDsCustomMsgInfo *custom_blob = (NvDsCustomMsgInfo *)frame_usermeta->user_meta_data;
            string msg = string((const char *)custom_blob->message, custom_blob->size);
            json_array_add_string_element(jArray, msg.c_str());
        }
    }
    if (json_array_get_length(jArray) > 0)
        json_object_set_array_member(rootObj, "customMessage", jArray);
    else
        json_array_unref(jArray);

    rootNode = json_node_new(JSON_NODE_OBJECT);
    json_node_set_object(rootNode, rootObj);

    message = json_to_string(rootNode, TRUE);
    json_node_free(rootNode);
    json_object_unref(rootObj);

    return message;
}

gchar *generate_dsmeta_message_minimal(void *privData, void *frameMeta)
{
    /*
  The JSON structure of the frame
  {
   "version": "4.0",
   "id": "frame-id",
   "@timestamp": "2018-04-11T04:59:59.828Z",
   "sensorId": "sensor-id",
   "objects": [
      ".......object-1 attributes...........",
      ".......object-2 attributes...........",
      ".......object-3 attributes..........."
    ]
  }
  */

    /*
  An example object with Vehicle object-type
  {
    "version": "4.0",
    "id": "frame-id",
    "@timestamp": "2018-04-11T04:59:59.828Z",
    "sensorId": "sensor-id",
    "objects": [
        "957|1834|150|1918|215|Vehicle|#|sedan|Bugatti|M|blue|CA 444|California|0.8",
        "..........."
    ]
  }
   */

    JsonNode *rootNode;
    JsonObject *jobject;
    JsonArray *jArray;
    stringstream ss;
    gchar *message = NULL;

    jArray = json_array_new();

    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)frameMeta;
    for (NvDsObjectMetaList *obj_l = frame_meta->obj_meta_list; obj_l; obj_l = obj_l->next) {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)obj_l->data;
        if (obj_meta == NULL) {
            // Ignore Null object.
            continue;
        }

        // bbox sub object
        float scaleW = (float)frame_meta->source_frame_width / (frame_meta->pipeline_width == 0)
                           ? 1
                           : frame_meta->pipeline_width;
        float scaleH = (float)frame_meta->source_frame_height / (frame_meta->pipeline_height == 0)
                           ? 1
                           : frame_meta->pipeline_height;

        float left = obj_meta->rect_params.left * scaleW;
        float top = obj_meta->rect_params.top * scaleH;
        float width = obj_meta->rect_params.width * scaleW;
        float height = obj_meta->rect_params.height * scaleH;

        ss.str("");
        ss.clear();
        ss << obj_meta->object_id << "|" << left << "|" << top << "|" << left + width << "|"
           << top + height << "|" << obj_meta->obj_label;

        if (g_list_length(obj_meta->classifier_meta_list) > 0) {
            ss << "|#";
            // Add classifiers for the object, if any
            for (NvDsClassifierMetaList *cl = obj_meta->classifier_meta_list; cl; cl = cl->next) {
                NvDsClassifierMeta *cl_meta = (NvDsClassifierMeta *)cl->data;
                for (NvDsLabelInfoList *ll = cl_meta->label_info_list; ll; ll = ll->next) {
                    NvDsLabelInfo *ll_meta = (NvDsLabelInfo *)ll->data;
                    ss << "|" << ll_meta->result_label;
                }
            }
            ss << "|" << obj_meta->confidence;
        }
        json_array_add_string_element(jArray, ss.str().c_str());
    }

    // generate timestamp
    char ts[MAX_TIME_STAMP_LEN + 1];
    generate_ts_rfc3339(ts, MAX_TIME_STAMP_LEN);

    // fetch sensor id
    string sensorId = "0";
    NvDsPayloadPriv *privObj = (NvDsPayloadPriv *)privData;
    auto idMap = privObj->sensorObj.find(frame_meta->source_id);
    if (idMap != privObj->sensorObj.end()) {
        NvDsSensorObject &obj = privObj->sensorObj[frame_meta->source_id];
        sensorId = obj.id;
    }

    jobject = json_object_new();
    json_object_set_string_member(jobject, "version", "4.0");
    json_object_set_int_member(jobject, "id", frame_meta->frame_num);
    json_object_set_string_member(jobject, "@timestamp", ts);
    json_object_set_string_member(jobject, "sensorId", sensorId.c_str());

    json_object_set_array_member(jobject, "objects", jArray);

    JsonArray *custMsgjArray = json_array_new();
    // Search for any custom message blob within frame usermeta list
    for (NvDsUserMetaList *l = frame_meta->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *frame_usermeta = (NvDsUserMeta *)l->data;
        if (frame_usermeta && frame_usermeta->base_meta.meta_type == NVDS_CUSTOM_MSG_BLOB) {
            NvDsCustomMsgInfo *custom_blob = (NvDsCustomMsgInfo *)frame_usermeta->user_meta_data;
            string msg = string((const char *)custom_blob->message, custom_blob->size);
            json_array_add_string_element(custMsgjArray, msg.c_str());
        }
    }
    if (json_array_get_length(custMsgjArray) > 0)
        json_object_set_array_member(jobject, "customMessage", custMsgjArray);
    else
        json_array_unref(custMsgjArray);

    rootNode = json_node_new(JSON_NODE_OBJECT);
    json_node_set_object(rootNode, jobject);

    message = json_to_string(rootNode, TRUE);
    json_node_free(rootNode);
    json_object_unref(jobject);

    return message;
}

/////////////////
/* Start Custom */
/////////////////

gchar *generate_dsmeta_message_minimal_custom(void *privData, void *frameMeta)
{
    JsonNode *rootNode;
    JsonObject *jobject;
    JsonArray *jArray;
    stringstream ss;
    gchar *message = NULL;
    JsonObject *objectObj;

    jArray = json_array_new();

    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)frameMeta;
    for (NvDsObjectMetaList *obj_l = frame_meta->obj_meta_list; obj_l; obj_l = obj_l->next) {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)obj_l->data;
        if (obj_meta == NULL || obj_meta->confidence <= 0) {
            // Ignore Null object.
            continue;
        }
        objectObj = generate_object_object_custom(privData, frame_meta, obj_meta);
        json_array_add_object_element(jArray, objectObj);
    }

    // generate timestamp
    char ts[MAX_TIME_STAMP_LEN + 1];
    generate_ts_rfc3339(ts, MAX_TIME_STAMP_LEN);

    jobject = json_object_new();
    json_object_set_string_member(jobject, "version", "4.0");
    // json_object_set_int_member(jobject, "frame_number", frame_meta->frame_num);
    json_object_set_int_member(jobject, "source_id", frame_meta->source_id);
    json_object_set_int_member(jobject, "num_frames_in_batch",
                               frame_meta->base_meta.batch_meta->num_frames_in_batch);
    json_object_set_int_member(jobject, "max_frames_in_batch",
                               frame_meta->base_meta.batch_meta->max_frames_in_batch);
    json_object_set_string_member(jobject, "msg_timestamp", ts);
    // json_object_set_double_member(jobject, "width", frame_meta->source_frame_width);
    // json_object_set_double_member(jobject, "height", frame_meta->source_frame_height);

    // std::cout << frame_meta->frame_num << "-" << frame_meta->bInferDone << std::endl;

    gchar *buf_pts = (gchar *)g_malloc0(MAX_TIME_STAMP_LEN);
    generate_ts_rfc3339_from_ts(buf_pts, MAX_TIME_STAMP_LEN, frame_meta->buf_pts);
    json_object_set_string_member(jobject, "buf_pts", buf_pts);

    gchar *ntp_timestamp = (gchar *)g_malloc0(MAX_TIME_STAMP_LEN);
    generate_ts_rfc3339_from_ts(ntp_timestamp, MAX_TIME_STAMP_LEN, frame_meta->ntp_timestamp);
    json_object_set_string_member(jobject, "ntp_timestamp", ntp_timestamp);

    json_object_set_array_member(jobject, "objects", jArray);

    // Search for any custom message blob within frame usermeta list
    for (NvDsUserMetaList *l = frame_meta->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *frame_usermeta = (NvDsUserMeta *)l->data;

        if (frame_usermeta->base_meta.meta_type == NVDS_CUSTOM_MSG_BLOB) {
            NvDsCustomMsgInfo *custom_blob = (NvDsCustomMsgInfo *)frame_usermeta->user_meta_data;
            string msg = string((const char *)custom_blob->message, custom_blob->size);

            GError *err = NULL;
            JsonNode *jnode = json_from_string(msg.c_str(), &err);

            JsonObject *jdata = json_node_get_object(jnode);
            GList *keys = json_object_get_members(jdata);

            for (GList *k = keys; k; k = g_list_next(k)) {
                JsonNode *node_value = json_object_get_member(jdata, (gchar *)k->data);
                if (json_object_has_member(jobject, (gchar *)k->data)) {
                    // TODO: Warnings
                }
                json_object_set_member(jobject, (gchar *)k->data, node_value);
            }
        }
    }

    rootNode = json_node_new(JSON_NODE_OBJECT);
    json_node_set_object(rootNode, jobject);

    message = json_to_string(rootNode, TRUE);
    json_node_free(rootNode);
    json_object_unref(jobject);

    return message;
}
////////////////
/* End Custom */
////////////////
