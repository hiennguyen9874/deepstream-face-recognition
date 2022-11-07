#include <cassert>
#include <cstring>
#include <iostream>

#include "nvdsinfer_custom_impl.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min, max) (MAX(MIN(a, max), min))

extern "C" bool NvDsInferParseCustomBatchedNMSTLT(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C" bool NvDsInferParseCustomEfficientNMSTLT(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C" bool NvDsInferParseCustomBatchedNMSTLT(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList) {
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

  // Must same as onnx config
  const int keep_top_k = outputLayersInfo[1].inferDims.d[0];

  const bool log_enable = false;

  if (log_enable) {
    std::cout << "keep cout: " << p_keep_count[0] << std::endl;
  }

  for (int i = 0; i < p_keep_count[0] && objectList.size() <= keep_top_k; i++) {
    if ((unsigned int)p_classes[i] >= detectionParams.numClassesConfigured)
      continue;

    if (p_scores[i] < 0.0)
      std::cout << "label/conf/ x/y x/y -- " << p_classes[i] << " "
                << p_scores[i] << " " << p_bboxes[4 * i] << " "
                << p_bboxes[4 * i + 1] << " " << p_bboxes[4 * i + 2] << " "
                << p_bboxes[4 * i + 3] << " " << std::endl;

    if (p_scores[i] < threshold)
      continue;

    if (log_enable) {
      std::cout << "label/conf/ x/y x/y -- " << p_classes[i] << " "
                << p_scores[i] << " " << p_bboxes[4 * i] << " "
                << p_bboxes[4 * i + 1] << " " << p_bboxes[4 * i + 2] << " "
                << p_bboxes[4 * i + 3] << " " << std::endl;
    }

    if (p_bboxes[4 * i + 2] < p_bboxes[4 * i] ||
        p_bboxes[4 * i + 3] < p_bboxes[4 * i + 1])
      continue;

    NvDsInferObjectDetectionInfo object;
    object.classId = (int)p_classes[i];
    object.detectionConfidence = p_scores[i];

    /* Clip object box co-ordinates to network resolution */
    object.left = CLIP(p_bboxes[4 * i], 0, networkInfo.width - 1);
    object.top = CLIP(p_bboxes[4 * i + 1], 0, networkInfo.height - 1);
    object.width =
        CLIP(p_bboxes[4 * i + 2], 0, networkInfo.width - 1) - object.left;
    object.height =
        CLIP(p_bboxes[4 * i + 3], 0, networkInfo.height - 1) - object.top;

    if (object.height < 0 || object.width < 0)
      continue;

    objectList.push_back(object);
  }
  return true;
}

extern "C" bool NvDsInferParseCustomEfficientNMSTLT(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList) {
  if (outputLayersInfo.size() != 4) {
    std::cerr << "Mismatch in the number of output buffers."
              << "Expected 4 output buffers, detected in the network :"
              << outputLayersInfo.size() << std::endl;
    return false;
  }

  /* Host memory for "EfficientNMS"
     EfficientNMS has 4 output bindings, the order is:
     keepCount, bboxes, scores, classes
  */

  int *p_keep_count = (int *)outputLayersInfo[0].buffer;
  float *p_bboxes = (float *)outputLayersInfo[1].buffer;
  float *p_scores = (float *)outputLayersInfo[2].buffer;
  int *p_classes = (int *)outputLayersInfo[3].buffer;

  const float threshold = detectionParams.perClassThreshold[0];

  // Must same as onnx config
  const int keep_top_k = outputLayersInfo[1].inferDims.d[0];

  const bool log_enable = false;

  if (log_enable) {
    std::cout << "keep cout: " << p_keep_count[0] << std::endl;
  }

  for (int i = 0; i < p_keep_count[0] && objectList.size() <= keep_top_k; i++) {
    if ((int)p_classes[i] >= detectionParams.numClassesConfigured)
      continue;

    if (p_scores[i] < 0.0)
      std::cout << "label/conf/ x/y x/y -- " << p_classes[i] << " "
                << p_scores[i] << " " << p_bboxes[4 * i] << " "
                << p_bboxes[4 * i + 1] << " " << p_bboxes[4 * i + 2] << " "
                << p_bboxes[4 * i + 3] << " " << std::endl;

    if (p_scores[i] < threshold)
      continue;

    if (log_enable) {
      std::cout << "label/conf/ x/y x/y -- " << p_classes[i] << " "
                << p_scores[i] << " " << p_bboxes[4 * i] << " "
                << p_bboxes[4 * i + 1] << " " << p_bboxes[4 * i + 2] << " "
                << p_bboxes[4 * i + 3] << " " << std::endl;
    }

    if (p_bboxes[4 * i + 2] < p_bboxes[4 * i] ||
        p_bboxes[4 * i + 3] < p_bboxes[4 * i + 1])
      continue;

    NvDsInferObjectDetectionInfo object;
    object.classId = (int)p_classes[i];
    object.detectionConfidence = p_scores[i];

    /* Clip object box co-ordinates to network resolution */
    object.left = CLIP(p_bboxes[4 * i], 0, networkInfo.width - 1);
    object.top = CLIP(p_bboxes[4 * i + 1], 0, networkInfo.height - 1);
    object.width =
        CLIP(p_bboxes[4 * i + 2], 0, networkInfo.width - 1) - object.left;
    object.height =
        CLIP(p_bboxes[4 * i + 3], 0, networkInfo.height - 1) - object.top;

    if (object.height < 0 || object.width < 0)
      continue;

    objectList.push_back(object);
  }
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomBatchedNMSTLT);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomEfficientNMSTLT);
