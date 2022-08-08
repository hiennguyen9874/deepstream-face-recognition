/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

#include "nvdsinfer_custom_impl.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min, max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

struct MrcnnRawDetection {
    float y1, x1, y2, x2, class_id, score;
};

/* This is a sample bounding box parsing function for the sample Resnet10
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomResnet(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                           NvDsInferNetworkInfo const &networkInfo,
                                           NvDsInferParseDetectionParams const &detectionParams,
                                           std::vector<NvDsInferObjectDetectionInfo> &objectList);

/* This is a sample bounding box parsing function for the tensorflow SSD models
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomTfSSD(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                          NvDsInferNetworkInfo const &networkInfo,
                                          NvDsInferParseDetectionParams const &detectionParams,
                                          std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C" bool NvDsInferParseCustomNMSTLT(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                           NvDsInferNetworkInfo const &networkInfo,
                                           NvDsInferParseDetectionParams const &detectionParams,
                                           std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C" bool NvDsInferParseCustomBatchedNMSTLT(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C" bool NvDsInferParseCustomMrcnnTLT(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList);

extern "C" bool NvDsInferParseCustomMrcnnTLTV2(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList);

extern "C" bool NvDsInferParseCustomResnet(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                           NvDsInferNetworkInfo const &networkInfo,
                                           NvDsInferParseDetectionParams const &detectionParams,
                                           std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    static NvDsInferDimsCHW covLayerDims;
    static NvDsInferDimsCHW bboxLayerDims;
    static int bboxLayerIndex = -1;
    static int covLayerIndex = -1;
    static bool classMismatchWarn = false;
    int numClassesToParse;

    /* Find the bbox layer */
    if (bboxLayerIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
            if (strcmp(outputLayersInfo[i].layerName, "conv2d_bbox") == 0) {
                bboxLayerIndex = i;
                getDimsCHWFromDims(bboxLayerDims, outputLayersInfo[i].inferDims);
                break;
            }
        }
        if (bboxLayerIndex == -1) {
            std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
            return false;
        }
    }

    /* Find the cov layer */
    if (covLayerIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
            if (strcmp(outputLayersInfo[i].layerName, "conv2d_cov/Sigmoid") == 0) {
                covLayerIndex = i;
                getDimsCHWFromDims(covLayerDims, outputLayersInfo[i].inferDims);
                break;
            }
        }
        if (covLayerIndex == -1) {
            std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
            return false;
        }
    }

    /* Warn in case of mismatch in number of classes */
    if (!classMismatchWarn) {
        if (covLayerDims.c != detectionParams.numClassesConfigured) {
            std::cerr << "WARNING: Num classes mismatch. Configured:"
                      << detectionParams.numClassesConfigured
                      << ", detected by network: " << covLayerDims.c << std::endl;
        }
        classMismatchWarn = true;
    }

    /* Calculate the number of classes to parse */
    numClassesToParse = MIN(covLayerDims.c, detectionParams.numClassesConfigured);

    int gridW = covLayerDims.w;
    int gridH = covLayerDims.h;
    int gridSize = gridW * gridH;
    float gcCentersX[gridW];
    float gcCentersY[gridH];
    float bboxNormX = 35.0;
    float bboxNormY = 35.0;
    float *outputCovBuf = (float *)outputLayersInfo[covLayerIndex].buffer;
    float *outputBboxBuf = (float *)outputLayersInfo[bboxLayerIndex].buffer;
    int strideX = DIVIDE_AND_ROUND_UP(networkInfo.width, bboxLayerDims.w);
    int strideY = DIVIDE_AND_ROUND_UP(networkInfo.height, bboxLayerDims.h);

    for (int i = 0; i < gridW; i++) {
        gcCentersX[i] = (float)(i * strideX + 0.5);
        gcCentersX[i] /= (float)bboxNormX;
    }
    for (int i = 0; i < gridH; i++) {
        gcCentersY[i] = (float)(i * strideY + 0.5);
        gcCentersY[i] /= (float)bboxNormY;
    }

    for (int c = 0; c < numClassesToParse; c++) {
        float *outputX1 = outputBboxBuf + (c * 4 * bboxLayerDims.h * bboxLayerDims.w);

        float *outputY1 = outputX1 + gridSize;
        float *outputX2 = outputY1 + gridSize;
        float *outputY2 = outputX2 + gridSize;

        float threshold = detectionParams.perClassPreclusterThreshold[c];
        for (int h = 0; h < gridH; h++) {
            for (int w = 0; w < gridW; w++) {
                int i = w + h * gridW;
                if (outputCovBuf[c * gridSize + i] >= threshold) {
                    NvDsInferObjectDetectionInfo object;
                    float rectX1f, rectY1f, rectX2f, rectY2f;

                    rectX1f = (outputX1[w + h * gridW] - gcCentersX[w]) * -bboxNormX;
                    rectY1f = (outputY1[w + h * gridW] - gcCentersY[h]) * -bboxNormY;
                    rectX2f = (outputX2[w + h * gridW] + gcCentersX[w]) * bboxNormX;
                    rectY2f = (outputY2[w + h * gridW] + gcCentersY[h]) * bboxNormY;

                    object.classId = c;
                    object.detectionConfidence = outputCovBuf[c * gridSize + i];

                    /* Clip object box co-ordinates to network resolution */
                    object.left = CLIP(rectX1f, 0, networkInfo.width - 1);
                    object.top = CLIP(rectY1f, 0, networkInfo.height - 1);
                    object.width = CLIP(rectX2f, 0, networkInfo.width - 1) - object.left + 1;
                    object.height = CLIP(rectY2f, 0, networkInfo.height - 1) - object.top + 1;

                    objectList.push_back(object);
                }
            }
        }
    }
    return true;
}

extern "C" bool NvDsInferParseCustomTfSSD(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                          NvDsInferNetworkInfo const &networkInfo,
                                          NvDsInferParseDetectionParams const &detectionParams,
                                          std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    auto layerFinder = [&outputLayersInfo](const std::string &name) -> const NvDsInferLayerInfo * {
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == FLOAT && (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *numDetectionLayer = layerFinder("num_detections");
    const NvDsInferLayerInfo *scoreLayer = layerFinder("detection_scores");
    const NvDsInferLayerInfo *classLayer = layerFinder("detection_classes");
    const NvDsInferLayerInfo *boxLayer = layerFinder("detection_boxes");
    if (!scoreLayer || !classLayer || !boxLayer) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }

    unsigned int numDetections = classLayer->inferDims.d[0];
    if (numDetectionLayer && numDetectionLayer->buffer) {
        numDetections = (int)((float *)numDetectionLayer->buffer)[0];
    }
    if (numDetections > classLayer->inferDims.d[0]) {
        numDetections = classLayer->inferDims.d[0];
    }
    numDetections = std::max<int>(0, numDetections);
    for (unsigned int i = 0; i < numDetections; ++i) {
        NvDsInferObjectDetectionInfo res;
        res.detectionConfidence = ((float *)scoreLayer->buffer)[i];
        res.classId = ((float *)classLayer->buffer)[i];
        if (res.classId >= detectionParams.perClassPreclusterThreshold.size() ||
            res.detectionConfidence < detectionParams.perClassPreclusterThreshold[res.classId]) {
            continue;
        }
        enum { y1, x1, y2, x2 };
        float rectX1f, rectY1f, rectX2f, rectY2f;
        rectX1f = ((float *)boxLayer->buffer)[i * 4 + x1] * networkInfo.width;
        rectY1f = ((float *)boxLayer->buffer)[i * 4 + y1] * networkInfo.height;
        rectX2f = ((float *)boxLayer->buffer)[i * 4 + x2] * networkInfo.width;
        ;
        rectY2f = ((float *)boxLayer->buffer)[i * 4 + y2] * networkInfo.height;
        rectX1f = CLIP(rectX1f, 0.0f, networkInfo.width - 1);
        rectX2f = CLIP(rectX2f, 0.0f, networkInfo.width - 1);
        rectY1f = CLIP(rectY1f, 0.0f, networkInfo.height - 1);
        rectY2f = CLIP(rectY2f, 0.0f, networkInfo.height - 1);
        if (rectX2f <= rectX1f || rectY2f <= rectY1f) {
            continue;
        }
        res.left = rectX1f;
        res.top = rectY1f;
        res.width = rectX2f - rectX1f;
        res.height = rectY2f - rectY1f;
        if (res.width && res.height) {
            objectList.emplace_back(res);
        }
    }

    return true;
}

extern "C" bool NvDsInferParseCustomMrcnnTLT(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList)
{
    auto layerFinder = [&outputLayersInfo](const std::string &name) -> const NvDsInferLayerInfo * {
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == FLOAT && (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *detectionLayer = layerFinder("generate_detections");
    const NvDsInferLayerInfo *maskLayer = layerFinder("mask_head/mask_fcn_logits/BiasAdd");

    if (!detectionLayer || !maskLayer) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }

    if (maskLayer->inferDims.numDims != 4U) {
        std::cerr << "Network output number of dims is : " << maskLayer->inferDims.numDims
                  << " expect is 4" << std::endl;
        return false;
    }

    const unsigned int det_max_instances = maskLayer->inferDims.d[0];
    const unsigned int num_classes = maskLayer->inferDims.d[1];
    if (num_classes != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << num_classes << std::endl;
    }
    const unsigned int mask_instance_height = maskLayer->inferDims.d[2];
    const unsigned int mask_instance_width = maskLayer->inferDims.d[3];

    auto out_det = reinterpret_cast<MrcnnRawDetection *>(detectionLayer->buffer);
    auto out_mask =
        reinterpret_cast<float(*)[mask_instance_width * mask_instance_height]>(maskLayer->buffer);

    for (auto i = 0U; i < det_max_instances; i++) {
        MrcnnRawDetection &rawDec = out_det[i];

        if (rawDec.score < detectionParams.perClassPreclusterThreshold[0])
            continue;

        NvDsInferInstanceMaskInfo obj;
        obj.left = CLIP(rawDec.x1, 0, networkInfo.width - 1);
        obj.top = CLIP(rawDec.y1, 0, networkInfo.height - 1);
        obj.width = CLIP(rawDec.x2, 0, networkInfo.width - 1) - rawDec.x1;
        obj.height = CLIP(rawDec.y2, 0, networkInfo.height - 1) - rawDec.y1;
        if (obj.width <= 0 || obj.height <= 0)
            continue;
        obj.classId = static_cast<int>(rawDec.class_id);
        obj.detectionConfidence = rawDec.score;

        obj.mask_size = sizeof(float) * mask_instance_width * mask_instance_height;
        obj.mask = new float[mask_instance_width * mask_instance_height];
        obj.mask_width = mask_instance_width;
        obj.mask_height = mask_instance_height;

        float *rawMask = reinterpret_cast<float *>(
            out_mask + i * detectionParams.numClassesConfigured + obj.classId);
        memcpy(obj.mask, rawMask, sizeof(float) * mask_instance_width * mask_instance_height);

        objectList.push_back(obj);
    }

    return true;
}

extern "C" bool NvDsInferParseCustomNMSTLT(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                           NvDsInferNetworkInfo const &networkInfo,
                                           NvDsInferParseDetectionParams const &detectionParams,
                                           std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    if (outputLayersInfo.size() != 2) {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 2 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    // Host memory for "nms" which has 2 output bindings:
    // the order is bboxes and keep_count
    float *out_nms = (float *)outputLayersInfo[0].buffer;
    int *p_keep_count = (int *)outputLayersInfo[1].buffer;
    const float threshold = detectionParams.perClassThreshold[0];

    float *det;

    for (int i = 0; i < p_keep_count[0]; i++) {
        det = out_nms + i * 7;

        // Output format for each detection is stored in the below order
        // [image_id, label, confidence, xmin, ymin, xmax, ymax]
        if (det[2] < threshold)
            continue;
        assert((unsigned int)det[1] < detectionParams.numClassesConfigured);

#if 0
        std::cout << "id/label/conf/ x/y x/y -- "
                  << det[0] << " " << det[1] << " " << det[2] << " "
                  << det[3] << " " << det[4] << " " << det[5] << " " << det[6] << std::endl;
#endif
        NvDsInferObjectDetectionInfo object;
        object.classId = (int)det[1];
        object.detectionConfidence = det[2];

        /* Clip object box co-ordinates to network resolution */
        object.left = CLIP(det[3] * networkInfo.width, 0, networkInfo.width - 1);
        object.top = CLIP(det[4] * networkInfo.height, 0, networkInfo.height - 1);
        object.width = CLIP((det[5] - det[3]) * networkInfo.width, 0, networkInfo.width - 1);
        object.height = CLIP((det[6] - det[4]) * networkInfo.height, 0, networkInfo.height - 1);

        objectList.push_back(object);
    }

    return true;
}

extern "C" bool NvDsInferParseCustomBatchedNMSTLT(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    if (outputLayersInfo.size() != 4) {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Host memory for "BatchedNMS"
       BatchedNMS has 4 output bindings, the order is:
       keepCount, bboxes, scores, classes
    */
    int *p_keep_count = (int *)outputLayersInfo[0].buffer;
    float *p_bboxes = (float *)outputLayersInfo[1].buffer;
    float *p_scores = (float *)outputLayersInfo[2].buffer;
    float *p_classes = (float *)outputLayersInfo[3].buffer;

    const float threshold = detectionParams.perClassThreshold[0];

    const int keep_top_k = 200;
    const char *log_enable = std::getenv("ENABLE_DEBUG");

    if (log_enable != NULL && std::stoi(log_enable)) {
        std::cout << "keep cout" << p_keep_count[0] << std::endl;
    }

    for (int i = 0; i < p_keep_count[0] && objectList.size() <= keep_top_k; i++) {
        if (p_scores[i] < threshold)
            continue;

        if (log_enable != NULL && std::stoi(log_enable)) {
            std::cout << "label/conf/ x/y x/y -- " << p_classes[i] << " " << p_scores[i] << " "
                      << p_bboxes[4 * i] << " " << p_bboxes[4 * i + 1] << " " << p_bboxes[4 * i + 2]
                      << " " << p_bboxes[4 * i + 3] << " " << std::endl;
        }

        if ((unsigned int)p_classes[i] >= detectionParams.numClassesConfigured)
            continue;
        if (p_bboxes[4 * i + 2] < p_bboxes[4 * i] || p_bboxes[4 * i + 3] < p_bboxes[4 * i + 1])
            continue;

        NvDsInferObjectDetectionInfo object;
        object.classId = (int)p_classes[i];
        object.detectionConfidence = p_scores[i];

        /* Clip object box co-ordinates to network resolution */
        object.left = CLIP(p_bboxes[4 * i] * networkInfo.width, 0, networkInfo.width - 1);
        object.top = CLIP(p_bboxes[4 * i + 1] * networkInfo.height, 0, networkInfo.height - 1);
        object.width =
            CLIP(p_bboxes[4 * i + 2] * networkInfo.width, 0, networkInfo.width - 1) - object.left;
        object.height =
            CLIP(p_bboxes[4 * i + 3] * networkInfo.height, 0, networkInfo.height - 1) - object.top;

        if (object.height < 0 || object.width < 0)
            continue;
        objectList.push_back(object);
    }
    return true;
}

extern "C" bool NvDsInferParseCustomMrcnnTLTV2(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList)
{
    auto layerFinder = [&outputLayersInfo](const std::string &name) -> const NvDsInferLayerInfo * {
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == FLOAT && (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *detectionLayer = layerFinder("generate_detections");
    const NvDsInferLayerInfo *maskLayer = layerFinder("mask_fcn_logits/BiasAdd");

    if (!detectionLayer || !maskLayer) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }

    if (maskLayer->inferDims.numDims != 4U) {
        std::cerr << "Network output number of dims is : " << maskLayer->inferDims.numDims
                  << " expect is 4" << std::endl;
        return false;
    }

    const unsigned int det_max_instances = maskLayer->inferDims.d[0];
    const unsigned int num_classes = maskLayer->inferDims.d[1];
    if (num_classes != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << num_classes << std::endl;
    }
    const unsigned int mask_instance_height = maskLayer->inferDims.d[2];
    const unsigned int mask_instance_width = maskLayer->inferDims.d[3];

    auto out_det = reinterpret_cast<MrcnnRawDetection *>(detectionLayer->buffer);
    auto out_mask =
        reinterpret_cast<float(*)[mask_instance_width * mask_instance_height]>(maskLayer->buffer);

    for (auto i = 0U; i < det_max_instances; i++) {
        MrcnnRawDetection &rawDec = out_det[i];

        if (rawDec.score < detectionParams.perClassPreclusterThreshold[0])
            continue;

        NvDsInferInstanceMaskInfo obj;
        obj.left = CLIP(rawDec.x1, 0, networkInfo.width - 1);
        obj.top = CLIP(rawDec.y1, 0, networkInfo.height - 1);
        obj.width = CLIP(rawDec.x2, 0, networkInfo.width - 1) - rawDec.x1;
        obj.height = CLIP(rawDec.y2, 0, networkInfo.height - 1) - rawDec.y1;
        if (obj.width <= 0 || obj.height <= 0)
            continue;
        obj.classId = static_cast<int>(rawDec.class_id);
        obj.detectionConfidence = rawDec.score;

        obj.mask_size = sizeof(float) * mask_instance_width * mask_instance_height;
        obj.mask = new float[mask_instance_width * mask_instance_height];
        obj.mask_width = mask_instance_width;
        obj.mask_height = mask_instance_height;

        float *rawMask = reinterpret_cast<float *>(
            out_mask + i * detectionParams.numClassesConfigured + obj.classId);
        memcpy(obj.mask, rawMask, sizeof(float) * mask_instance_width * mask_instance_height);

        objectList.push_back(obj);
    }

    return true;
}
/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomResnet);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomTfSSD);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomNMSTLT);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomBatchedNMSTLT);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomMrcnnTLT);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomMrcnnTLTV2);
