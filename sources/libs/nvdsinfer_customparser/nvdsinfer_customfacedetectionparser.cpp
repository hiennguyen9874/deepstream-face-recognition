#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "nvdsinfer_custom_impl.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min, max) (MAX(MIN(a, max), min))

const static int kNUM_CLASSES = 1;
const static float kNMS_THRESH = 0.45;
const static float kLANDMARK_THRES = 0.5;

static std::vector<NvDsInferFaceDetectionLandmarkInfo> nonMaximumSuppression(
    const float nmsThresh,
    std::vector<NvDsInferFaceDetectionLandmarkInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min) {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto computeIoU = [&overlap1D](NvDsInferFaceDetectionLandmarkInfo &bbox1,
                                   NvDsInferFaceDetectionLandmarkInfo &bbox2) -> float {
        float overlapX =
            overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
        float overlapY =
            overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
        float area1 = (bbox1.width) * (bbox1.height);
        float area2 = (bbox2.width) * (bbox2.height);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const NvDsInferFaceDetectionLandmarkInfo &b1,
                        const NvDsInferFaceDetectionLandmarkInfo &b2) {
                         return b1.detectionConfidence > b2.detectionConfidence;
                     });

    std::vector<NvDsInferFaceDetectionLandmarkInfo> out;
    for (auto i : binfo) {
        bool keep = true;
        for (auto j : out) {
            if (keep) {
                float overlap = computeIoU(i, j);
                keep = overlap <= nmsThresh;
            } else
                break;
        }
        if (keep)
            out.push_back(i);
    }
    return out;
}

static std::vector<NvDsInferFaceDetectionLandmarkInfo> nmsAllClasses(
    const float nmsThresh,
    std::vector<NvDsInferFaceDetectionLandmarkInfo> &binfo,
    const uint numClasses)
{
    std::vector<NvDsInferFaceDetectionLandmarkInfo> result;

    std::vector<std::vector<NvDsInferFaceDetectionLandmarkInfo>> splitBoxes(numClasses);

    for (auto &box : binfo) {
        splitBoxes.at(box.classId).push_back(box);
    }

    for (auto &boxes : splitBoxes) {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }
    return result;
}

static NvDsInferFaceDetectionLandmarkInfo convertBBox(const float &bx,
                                                      const float &by,
                                                      const float &bw,
                                                      const float &bh,
                                                      const uint &netW,
                                                      const uint &netH)
{
    NvDsInferFaceDetectionLandmarkInfo b;

    float x0 = bx - bw / 2;
    float y0 = by - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;

    x0 = CLIP(x0, 0, netW);
    y0 = CLIP(y0, 0, netH);
    x1 = CLIP(x1, 0, netW);
    y1 = CLIP(y1, 0, netH);

    b.left = x0;
    b.width = CLIP(x1 - x0, 0, netW);
    b.top = y0;
    b.height = CLIP(y1 - y0, 0, netH);

    return b;
}

static void addBBoxProposal(const float bx,
                            const float by,
                            const float bw,
                            const float bh,
                            const uint &netW,
                            const uint &netH,
                            const int label,
                            const float prob,
                            const float *landmark,
                            const unsigned int num_landmark,
                            const unsigned int landmark_size,
                            std::vector<NvDsInferFaceDetectionLandmarkInfo> &binfo)
{
    NvDsInferFaceDetectionLandmarkInfo bbi = convertBBox(bx, by, bw, bh, netW, netH);

    if (bbi.width < 1 || bbi.height < 1)
        return;

    bbi.detectionConfidence = prob;
    bbi.classId = label;

    bbi.num_landmark = num_landmark;
    bbi.landmark_size = landmark_size;

    bbi.landmark = new float[num_landmark * 2];

    memcpy(bbi.landmark, landmark, landmark_size);

    binfo.push_back(bbi);
}

static std::vector<NvDsInferFaceDetectionLandmarkInfo> decodeYoloTensor(const float *detections,
                                                                        const uint numOutputClasses,
                                                                        const uint numAnchors,
                                                                        const uint &netW,
                                                                        const uint &netH,
                                                                        const float confThresh)
{
    std::vector<NvDsInferFaceDetectionLandmarkInfo> binfo;

    for (int anchor_idx = 0; anchor_idx < numAnchors; anchor_idx++) {
        const int basic_pos = anchor_idx * (numOutputClasses + 5 + 10);

        float x_center = detections[basic_pos + 0];
        float y_center = detections[basic_pos + 1];
        float w = detections[basic_pos + 2];
        float h = detections[basic_pos + 3];
        float box_objectness = detections[basic_pos + 4];

        const unsigned int num_landmark = 5;
        const unsigned int landmark_size = sizeof(float) * num_landmark * 2;

        float *landmark =
            new float[num_landmark * 2]{detections[basic_pos + numOutputClasses + 5],
                                        detections[basic_pos + numOutputClasses + 6],
                                        detections[basic_pos + numOutputClasses + 7],
                                        detections[basic_pos + numOutputClasses + 8],
                                        detections[basic_pos + numOutputClasses + 9],
                                        detections[basic_pos + numOutputClasses + 10],
                                        detections[basic_pos + numOutputClasses + 11],
                                        detections[basic_pos + numOutputClasses + 12],
                                        detections[basic_pos + numOutputClasses + 13],
                                        detections[basic_pos + numOutputClasses + 14]};

        if (box_objectness > confThresh) {
            for (int class_idx = 0; class_idx < numOutputClasses; class_idx++) {
                float box_cls_score = detections[basic_pos + 5 + class_idx];
                float box_prob = box_objectness * box_cls_score;

                if (box_prob > confThresh) {
                    addBBoxProposal(x_center, y_center, w, h, netW, netH, class_idx, box_prob,
                                    landmark, num_landmark, landmark_size, binfo);
                }
            }
        }
    }

    return binfo;
}

extern "C" bool NvDsInferParseCustomYoloFaceDetection(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferFaceDetectionLandmarkInfo> &objectList)
{
    const uint numClasses = kNUM_CLASSES;

    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
        ;
        return false;
    }

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];

    assert(layer.inferDims.numDims == 2);

    const uint numAnchors = layer.inferDims.d[0];

    if (numClasses != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch. Configured: "
                  << detectionParams.numClassesConfigured << ", detected by network: " << numClasses
                  << std::endl;
    }

    std::vector<NvDsInferFaceDetectionLandmarkInfo> objects;

    objects =
        decodeYoloTensor((const float *)(layer.buffer), numClasses, numAnchors, networkInfo.width,
                         networkInfo.height, detectionParams.perClassThreshold[0]);

    objectList = nmsAllClasses(kNMS_THRESH, objects, numClasses);

    // objectList = objects;

    return true;
}

CHECK_CUSTOM_FACE_DETECTION_LANDMARK_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloFaceDetection);
