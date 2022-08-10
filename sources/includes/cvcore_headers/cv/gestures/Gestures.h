/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NV_GESTURES_H_
#define NV_GESTURES_H_

#include <cuda_runtime.h>
#include <cv/core/Array.h>
#include <cv/core/BBox.h>
#include <cv/core/Core.h>
#include <cv/core/Model.h>
#include <cv/core/Tensor.h>

#include <memory>

namespace cvcore {
namespace gestures {

/**
 * Describes the parameters for filtering the gesture detections.
 */
struct GesturesPostProcessorParams {
    /**< Number of gestures */
    uint32_t numGestures;
    /**< Smoothing window size for filtering the detections. Set to 1 for no filtering */
    size_t gestureHistorySize;
    /**< Threshold for refining. Refers to the maximum number of detections in the window size. */
    size_t gestureHistoryThreshold;
};

/**
 *  Default parameters for the preprocessing pipeline.
 */
CVCORE_API extern const ImagePreProcessingParams defaultPreProcessorParams;

/**
 *  Default parameters to describe the input expected for the model.
 */
CVCORE_API extern const ModelInputParams defaultModelInputParams;

/**
 *  Default parameters to describe the model inference parameters.
 */
CVCORE_API extern const ModelInferenceParams defaultInferenceParams;

/**
 *  Default parameters for the post processing pipeline.
 */
CVCORE_API extern const GesturesPostProcessorParams defaultPostProcessorParams;

/*
 * Interface for running pre-processing on gestures model.
 */
class CVCORE_API GesturesPreProcessor {
public:
    /**
     * Default constructor is deleted
     */
    GesturesPreProcessor() = delete;

    /**
     * Constructor of GesturesPreProcessor.
     * @param preProcessorParams image pre-processing parameters.
     * @param modelInputParams model paramters for network.
     */
    GesturesPreProcessor(const ImagePreProcessingParams &preProcessorParams,
                         const ModelInputParams &modelInputParams);

    /**
     * Destructor of GesturesPreProcessor.
     */
    ~GesturesPreProcessor();

    /**
     * Main interface to run pre-processing.
     * @param stream cuda stream.
     */

    void execute(Tensor<NCHW, C3, F32> &output,
                 const Tensor<NHWC, C3, U8> &input,
                 const Array<BBox> &inputBBoxes,
                 cudaStream_t stream = 0);

private:
    /**
     * Implementation of GesturesPreProcessor.
     */
    struct GesturesPreProcessorImpl;
    std::unique_ptr<GesturesPreProcessorImpl> m_pImpl;
};

/**
 * Gestures parameters and implementation
 */
class CVCORE_API Gestures {
public:
    /**
     * Top gestures detected in order of likelihood.
     */
    static constexpr size_t TOP_GESTURES = 10;

    using GesturesLikelihood = ArrayN<std::pair<uint32_t, float>, TOP_GESTURES>;

    /**
     * Constructor for Gestures.
     * @param params Gestures config parameters.
     */
    Gestures(const ImagePreProcessingParams &imgparams,
             const ModelInputParams &modelParams,
             const ModelInferenceParams &modelInferParams,
             const GesturesPostProcessorParams &postProcessParams);

    /**
     * Default constructor not supported.
     */
    Gestures() = delete;

    /**
     * Destructor for Gestures.
     */
    ~Gestures();

    /**
     * Inference function for a given BGR image
     * @param gestures Array of gestures detected represented as idx, score ranked
     * based on highest score.
     * @param input RGB/BGR Interleaved image (CPU/GPU Input Tensor supported)
     * @param inputBBox Bounding box of hand
     * @param stream Cuda stream
     */
    void execute(Array<GesturesLikelihood> &gestures,
                 const Tensor<NHWC, C3, U8> &input,
                 const Array<BBox> &inputBBox,
                 cudaStream_t stream = 0);

private:
    struct GesturesImpl;
    std::unique_ptr<GesturesImpl> m_pImpl;
};

/**
 * Interface for running post-processing on gestures network.
 */
class CVCORE_API GesturesPostProcessor {
public:
    GesturesPostProcessor() = delete;
    /**
     * Constructor of GesturesPostProcessor.
     * @param modelInputParams model paramters for network.
     */
    GesturesPostProcessor(const ModelInputParams &inputParams,
                          const GesturesPostProcessorParams &postProcessParams);

    /**
     * Destructor of GesturesPostProcessor.
     */
    ~GesturesPostProcessor();

    /**
     * Allocate staging CPU buffers (used when inputs are GPU Tensors).
     */
    void allocateStagingBuffers();

    /**
     * Main interface to run post-processing for batch input.
     * @param gestures Array of gestures detected represented as pair<index, score> ranked by score
     * based on highest score.
     * @param input Raw gesture output from the output layer of gestures network.
     * @param stream Cuda stream
     */
    void execute(Array<Gestures::GesturesLikelihood> &gestures,
                 const Tensor<CL, CX, F32> &input,
                 cudaStream_t stream = 0);

private:
    /**
     * Implementation of Pose2DPostProcessor.
     */
    struct GesturesPostProcessorImpl;
    std::unique_ptr<GesturesPostProcessorImpl> m_pImpl;
};

} // namespace gestures
} // namespace cvcore
#endif
