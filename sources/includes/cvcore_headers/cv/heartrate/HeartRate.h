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

#ifndef CVCORE_HEARTRATE_H_
#define CVCORE_HEARTRATE_H_

#include <cuda_runtime.h>
#include <cv/core/Array.h>
#include <cv/core/BBox.h>
#include <cv/core/Model.h>
#include <cv/core/Tensor.h>

#include <memory>

namespace cvcore {
namespace heartrate {

/**
 * Interface for loading and running HeartRate.
 */
class HeartRate {
public:
    /**
     * Default Image Processing Params for HeartRate.
     */
    static const ImagePreProcessingParams defaultPreProcessorParams;

    /**
     * Default Model Input Params for HeartRate.
     */
    static const ModelInputParams defaultModelInputParams;

    /**
     * Default inference Params for HeartRate.
     */
    static const ModelInferenceParams defaultInferenceParams;

    /**
     * HeartRate extra params
     */
    struct HeartRateParams {
        std::size_t cameraFps;          // FPS of the camera to map heart rate to time
        std::size_t heartRateMax;       // Maximum valid heart rate
        std::size_t heartRateMin;       // Minimum valid heart rate
        std::size_t estimationInterval; // The minimum number of seconds of inference
                                        //     points to caluclate FFT heart rate
        std::size_t warmUpInterval;     // The minimum number of seconds of inferemence
                                        //     samples to begin calculation
        std::size_t fftWindow;          // The size of the FFT lookback
        std::int32_t roiSnrThresh;      // The minimum SNR on the FFT window
        std::size_t maxConfidence;      // Maximum clamp confidence of the the inference
        std::size_t minConfidence;      // Minimum FFT confidence to provide a heart rate
        std::size_t validFpsGap;        // Maximum time variation in heart rate estimates
        float iouThresh;                // IoU threshold for determining if a persistent
                                        //     object is present in frame
    };
    static const HeartRateParams defaultExtraParams;

    /**
     * Removing the default constructor for HeartRate.
     */
    HeartRate() = delete;

    /**
     * Constructor for HeartRate.
     * @param preProcessorParams Image preprocessing parameters.
     * @param modelInputParams Model input parameters.
     * @param inferenceParams Inference parameters for the model.
     * @param extraParams Model parameters unique to this model.
     */
    HeartRate(const ImagePreProcessingParams &preProcessorParams,
              const ModelInputParams &modelInputParams,
              const ModelInferenceParams &inferenceParams,
              const HeartRateParams &extraParams);

    /**
     * Destructor for HeartRate.
     */
    ~HeartRate();

    /**
     * Running HeartRate for a given image.
     * @param heartRate output heartrate estimation for each image in the batch.
     * @param faceImage input batch of frame images containing faces
     * @param faceBBox input batch of bounding boxes, one per frame per face.
     * @param stream Cuda stream
     */
    void execute(Array<float> &heartRate,
                 const Tensor<NCHW, C3, U8> &faceImage,
                 const Array<BBox> &faceBBox,
                 cudaStream_t stream = 0);

    void execute(Array<float> &heartRate,
                 const Tensor<NHWC, C3, U8> &faceImage,
                 const Array<BBox> &faceBBox,
                 cudaStream_t stream = 0);

private:
    struct HeartRateImpl;

    std::unique_ptr<HeartRateImpl> m_pImpl;
};

/**
 * Interface for running pre-processing for HeartRate.
 */
class HeartRatePreProcessor {
public:
    /**
     * Removing the default constructor for HeartRatePreProcessor.
     */
    HeartRatePreProcessor() = delete;

    /**
     * Constructor for HeartRatePreProcessor.
     * @param preProcessorParams Image preprocessing parameters.
     * @param modelInputParams Model input parameters.
     * @param extraParams Model parameters unique to this model.
     */
    HeartRatePreProcessor(const ImagePreProcessingParams &preProcessorParams,
                          const ModelInputParams &modelInputParams,
                          const HeartRate::HeartRateParams &extraParams);

    /**
     * Destructor for HeartRatePreProcessor.
     */
    ~HeartRatePreProcessor();

    /**
     * Running preprocessing for a given face image and bounding box.
     * @param preProcessedFaceImage output preprocessed batch face image.
     * @param preProcessedFaceMotion output motion image of the last two frame.
     * @param entityInFrame vector of booleans determine if persistent object in frame.
     * @param faceImage input batch of raw face images.
     * @param faceBBox input batch of face bounding boxes.
     * @param stream Cuda stream
     */
    void execute(Tensor<NCHW, C3, F32> &preProcessedFaceImage,
                 Tensor<NCHW, C3, F32> &preProcessedFaceMotion,
                 Array<bool> &entityInFrame,
                 const Tensor<NCHW, C3, U8> &faceImage,
                 const Array<BBox> &faceBBox,
                 cudaStream_t stream = 0);

    void execute(Tensor<NCHW, C3, F32> &preProcessedFaceImage,
                 Tensor<NCHW, C3, F32> &preProcessedFaceMotion,
                 Array<bool> &entityInFrame,
                 const Tensor<NHWC, C3, U8> &faceImage,
                 const Array<BBox> &faceBBox,
                 cudaStream_t stream = 0);

private:
    struct HeartRatePreProcessorImpl;

    std::unique_ptr<HeartRatePreProcessorImpl> m_pImpl;
};

/**
 * Interface for running post-processing for HeartRate.
 */
class HeartRatePostProcessor {
public:
    /**
     * Removing the default constructor for HeartRatePostProcessor.
     */
    HeartRatePostProcessor() = delete;

    /**
     * Constructor for HeartRatePostProcessor.
     * @param modelInputParams Model input parameters.
     * @param extraParams Model parameters unique to this model.
     */
    HeartRatePostProcessor(const ModelInputParams &modelInputParams,
                           const HeartRate::HeartRateParams &extraParams);

    /**
     * Destructor for HeartRatePostProcessor.
     */
    ~HeartRatePostProcessor();

    /**
     * Running postprocessing for a given heart rate inference history.
     * @param heartRate output heartrate calculated using signal processing on the inference
     * history.
     * @param rawHeartRate the current heart rate inference based on last two frames.
     * @param entityInFrame vector of booleans determine if persistent object in frame.
     * @param stream Cuda stream
     */
    void execute(Array<float> &heartRate,
                 const Tensor<CL, CX, F32> &rawHeartRate,
                 const Array<bool> &entityInFrame,
                 cudaStream_t stream = 0);

private:
    struct HeartRatePostProcessorImpl;

    std::unique_ptr<HeartRatePostProcessorImpl> m_pImpl;
};

} // namespace heartrate
} // namespace cvcore

#endif // CVCORE_HEARTRATE_H_
