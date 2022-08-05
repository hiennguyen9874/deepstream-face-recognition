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

#ifndef CVCORE_GAZENET_H_
#define CVCORE_GAZENET_H_

#include <cuda_runtime.h>
#include <cv/core/Array.h>
#include <cv/core/BBox.h>
#include <cv/core/Core.h>
#include <cv/core/Image.h>
#include <cv/core/MathTypes.h>
#include <cv/core/Model.h>
#include <cv/core/Tensor.h>

#include <memory>

namespace cvcore {
namespace gazenet {

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
 * Describes the type of model for Gazenet.
 */
enum class FeatureType { FACEGRID, LANDMARKS };

/**
 * Interface for running pre-processing on GazeNet.
 */
class CVCORE_API GazeNetPreProcessor {
public:
    /**
     * Number of landmarks needed for the model.
     */
    static constexpr size_t NUM_LANDMARKS = 68;

    /**
     * Scaling factor for face.
     */
    static constexpr float BBOX_FACE_SCALE = 1.3f;

    /**
     * Scaling factor for eyes.
     */
    static constexpr float BBOX_EYE_SCALE = 1.8f;

    /**
     * Default landmarks mean values for the model.
     */
    static const float DEFAULT_LANDMARKS_MEAN[];

    /**
     * Default landmarks standard deviation values for the model.
     */
    static const float DEFAULT_LANDMARKS_STD[];

    /**
     * Default constructor is deleted
     */
    GazeNetPreProcessor() = delete;

    /**
     * Constructor of GazeNetPreProcessor.
     * @param preProcessorParams image pre-processing parameters.
     * @param modelInputParams model paramters for network.
     */
    GazeNetPreProcessor(const ImagePreProcessingParams &preProcessorParams,
                        const ModelInputParams &modelInputParams);

    /**
     * Destructor of GazeNetPreProcessor.
     */
    ~GazeNetPreProcessor();

    /**
     * Set the mean and standard deviation of the landmarks for the model.
     * @param landmarksMean mean of the landmarks.
     * @param landmarksStd standard deviation of the landmarks.
     */
    void setLandmarksMeanAndVariance(const Array<Vector2f> &landmarksMean,
                                     const Array<Vector2f> &landmarksStd);

    /**
     * Main interface to run pre-processing.
     * @param outputFace output tensor for face region.
     * @param outputLeft output tensor for left eye.
     * @param outputRight output tensor for right eye.
     * @param outputFeatures output normalized landmarks or facegrid.
     * @param inputImage input image tensor.
     * @param inputBBox input BBox of the face.
     * @param inputLandmarks input raw facial landmarks.
     * @param type whether to use facegrid model or landmarks model.
     * @param stream cuda stream.
     */
    void execute(Tensor<NHWC, C1, F32> &outputFace,
                 Tensor<NHWC, C1, F32> &outputLeft,
                 Tensor<NHWC, C1, F32> &outputRight,
                 Tensor<CL, CX, F32> &outputFeatures,
                 const Tensor<NHWC, C3, U8> &inputImage,
                 const Array<BBox> &inputBBox,
                 const Array<ArrayN<Vector2f, GazeNetPreProcessor::NUM_LANDMARKS>> &inputLandmarks,
                 FeatureType type = FeatureType::FACEGRID,
                 cudaStream_t stream = 0);

private:
    /**
     * Implementation of GazeNetPreProcessor.
     */
    struct GazeNetPreProcessorImpl;
    std::unique_ptr<GazeNetPreProcessorImpl> m_pImpl;
};

/**
 * Interface for loading and running GazeNet.
 */
class CVCORE_API GazeNet {
public:
    /**
     * Number of elements in network output.
     */
    static constexpr size_t OUTPUT_SIZE = 5;

    /**
     * Width of facegrid.
     */
    static constexpr size_t FACEGRID_WIDTH = 25;

    /**
     * Height of facegrid.
     */
    static constexpr size_t FACEGRID_HEIGHT = 25;

    /**
     * Starting index of left eye region.
     */
    static constexpr size_t LANDMARKS_EYE_LEFT_BEGIN = 42;

    /**
     * End index of left eye region.
     */
    static constexpr size_t LANDMARKS_EYE_LEFT_END = 48;

    /**
     * Starting index of right eye region.
     */
    static constexpr size_t LANDMARKS_EYE_RIGHT_BEGIN = 36;

    /**
     * End index of right eye region.
     */
    static constexpr size_t LANDMARKS_EYE_RIGHT_END = 42;

    /**
     * Default constructor is deleted.
     */
    GazeNet() = delete;

    /**
     * Constructor of GazeNet.
     * @param preProcessorParams custom pre-processor params.
     * @param modelInputParams custom model input params for the network.
     * @param inferenceParams custom inference params for the network.
     */
    GazeNet(const ImagePreProcessingParams &preProcessorParams,
            const ModelInputParams &modelInputParams,
            const ModelInferenceParams &inferenceParams);

    /**
     * Destructor of GazeNet.
     */
    ~GazeNet();

    /**
     * Set the mean and standard deviation of the landmarks for the model.
     * @param landmarksMean mean of the landmarks.
     * @param landmarksStd standard deviation of the landmarks.
     */
    void setLandmarksMeanAndVariance(const Array<Vector2f> &landmarksMean,
                                     const Array<Vector2f> &landmarksStd);

    /**
     * Main interface to run inference.
     * @param outputGaze output gaze results.
     * @param inputImage input RGB/BGR interleaved image.
     * @param inputBBox input Bounding box for face region.
     * @param inputLandmarks input landmarks coordinates of the face.
     * @param type whether to use facegrid model or landmarks model.
     * @param stream cuda stream.
     */
    void execute(Array<ArrayN<float, GazeNet::OUTPUT_SIZE>> &outputGaze,
                 const Tensor<NHWC, C3, U8> &inputImage,
                 const Array<BBox> &inputBBox,
                 const Array<ArrayN<Vector2f, GazeNetPreProcessor::NUM_LANDMARKS>> &inputLandmarks,
                 FeatureType type = FeatureType::FACEGRID,
                 cudaStream_t stream = 0);

private:
    /**
     * Implementation of GazeNet.
     */
    struct GazeNetImpl;
    std::unique_ptr<GazeNetImpl> m_pImpl;
};

} // namespace gazenet
} // namespace cvcore

#endif
