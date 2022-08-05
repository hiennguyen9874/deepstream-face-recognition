/*###############################################################################
#
# Copyright(c) 2021 NVIDIA CORPORATION. All Rights Reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
###############################################################################*/

#ifndef CVCORE_EMOTIONS_H_
#define CVCORE_EMOTIONS_H_

#include <cuda_runtime.h>
#include <cv/core/Array.h>
#include <cv/core/BBox.h>
#include <cv/core/MathTypes.h>
#include <cv/core/Model.h>
#include <cv/core/Tensor.h>

#include <memory>

namespace cvcore {
namespace emotions {

/**
 * Default Image Processing Params for Emotions.
 */
extern const ImagePreProcessingParams defaultPreProcessorParams;

/**
 * Default Model Input Params for Emotions.
 */
extern const ModelInputParams defaultModelInputParams;

/**
 * Default inference Params for Emotions.
 */
extern const ModelInferenceParams defaultInferenceParams;

/**
 * Interface for loading and running Emotions.
 */
class Emotions {
public:
    /**
     * Number of input facial landmarks.
     */
    static constexpr std::size_t NUM_FACIAL_LANDMARKS = 68;

    /**
     * Top emotions classified in order of likelihood.
     */
    static constexpr std::size_t TOP_EMOTIONS = 10;

    /**
     * Removing the default constructor for Emotions.
     */
    Emotions() = delete;

    /**
     * Constructor for Emotions.
     * @param preProcessorParams Image preprocessing parameters.
     * @param modelInputParams Model input parameters.
     * @param inferenceParams Inference parameters for the model.
     * @param numEmotions number of the output emotions
     */
    Emotions(const ImagePreProcessingParams &preProcessorParams,
             const ModelInputParams &modelInputParams,
             const ModelInferenceParams &inferenceParams,
             size_t numEmotions = 6);

    /**
     * Destructor for Emotions.
     */
    ~Emotions();

    /**
     * Running Emotions for a given image.
     * @param emotionLikelihoods output emotions likelihood vector for each image in the batch.
     * @param topEmotions the top ranked emotions for each image in the batch
     * @param inputLandmarks input facial landmarks vector.
     * @param stream Cuda stream
     */
    void execute(Array<ArrayN<float, Emotions::TOP_EMOTIONS>> &emotionLikelihoods,
                 Array<ArrayN<size_t, Emotions::TOP_EMOTIONS>> &topEmotions,
                 const Array<ArrayN<Vector2f, NUM_FACIAL_LANDMARKS>> &inputLandmarks,
                 cudaStream_t stream = 0);

private:
    struct EmotionsImpl;

    std::unique_ptr<EmotionsImpl> m_pImpl;
};

/**
 * Interface for running pre-processing for Emotions.
 */
class EmotionsPreProcessor {
public:
    /**
     * Removing the default constructor for EmotionsPreProcessor.
     */
    EmotionsPreProcessor() = delete;

    /**
     * Constructor for EmotionsPreProcessor.
     * @param preProcessorParams Image preprocessing parameters.
     * @param modelInputParams Model input parameters.
     */
    EmotionsPreProcessor(const ImagePreProcessingParams &preProcessorParams,
                         const ModelInputParams &modelInputParams);

    /**
     * Destructor for EmotionsPreProcessor.
     */
    ~EmotionsPreProcessor();

    /**
     * Running preprocessing for a given set of facial landmarks.
     * @param preProcessedLandmarkBatch input landmarks vector in tensor format.
     * @param inputLandmarks input landmarks vector in array format.
     * @param stream Cuda stream
     */
    void execute(Tensor<CL, CX, F32> &preProcessedLandmarkBatch,
                 const Array<ArrayN<Vector2f, Emotions::NUM_FACIAL_LANDMARKS>> &inputLandmarks,
                 cudaStream_t stream = 0);

private:
    struct EmotionsPreProcessorImpl;

    std::unique_ptr<EmotionsPreProcessorImpl> m_pImpl;
};

/**
 * Interface for running post-processing for Emotions.
 */
class EmotionsPostProcessor {
public:
    /**
     * Removing the default constructor for EmotionsPostProcessor.
     */
    EmotionsPostProcessor() = delete;

    /**
     * Constructor for EmotionsPostProcessor.
     * @param modelInputParams Model input parameters.
     * @param numEmotions number of the output emotions
     */
    EmotionsPostProcessor(const ModelInputParams &modelInputParams, size_t numEmotions);

    /**
     * Destructor for EmotionsPostProcessor.
     */
    ~EmotionsPostProcessor();

    /**
     * Running postprocessing for a given set of emotion likelihoods.
     * @param emotionLikelihoods output emotions likelihood vector for each image in the batch.
     * @param topEmotions the top ranked emotions for each image in the batch
     * @param emotionsRaw input emotions vector in tensor format.
     * @param stream Cuda stream
     */
    void execute(Array<ArrayN<float, Emotions::TOP_EMOTIONS>> &emotionLikelihoods,
                 Array<ArrayN<std::size_t, Emotions::TOP_EMOTIONS>> &topEmotions,
                 const Tensor<CL, CX, F32> &emotionsRaw,
                 cudaStream_t stream = 0);

private:
    struct EmotionsPostProcessorImpl;

    std::unique_ptr<EmotionsPostProcessorImpl> m_pImpl;
};

} // namespace emotions
} // namespace cvcore

#endif // CVCORE_EMOTIONS_H_
