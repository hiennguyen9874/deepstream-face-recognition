/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CVCORE_MODEL_H
#define CVCORE_MODEL_H

#include <vector>

#include "Image.h"

namespace cvcore {

/**
 * Struct to describe input type required by the model
 */
struct ModelInputParams {
    size_t maxBatchSize;      /**< maxbatchSize supported by network*/
    size_t inputLayerWidth;   /**< Input layer width */
    size_t inputLayerHeight;  /**< Input layer Height */
    ImageType modelInputType; /**< Input Layout type */
};

/**
 * Struct to describe the model
 */
struct ModelInferenceParams {
    std::string engineFilePath;            /**< Engine file path. */
    std::vector<std::string> inputLayers;  /**< names of input layers. */
    std::vector<std::string> outputLayers; /**< names of output layers. */
};

} // namespace cvcore

#endif // CVCORE_MODEL_H
