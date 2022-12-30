/*###############################################################################
#
# Copyright(c) 2020 NVIDIA CORPORATION. All Rights Reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
###############################################################################*/

#ifndef CVCORE_FACIALLANDMARKS_H_
#define CVCORE_FACIALLANDMARKS_H_

#include <cuda_runtime.h>
#include <cv/core/Array.h>
#include <cv/core/BBox.h>
#include <cv/core/Core.h>
#include <cv/core/MathTypes.h>
#include <cv/core/Model.h>
#include <cv/core/Tensor.h>

#include <memory>

namespace cvcore {
namespace faciallandmarks {

/**
 * Default Image Processing Params for Facial Landmarks.
 */
CVCORE_API extern const ImagePreProcessingParams defaultPreProcessorParams;

/**
 * Default Model Input Params for FacialLandmarks.
 */
CVCORE_API extern const ModelInputParams defaultModelInputParams;

/**
 * Default inference Params for FacialLandmarks.
 */
CVCORE_API extern const ModelInferenceParams defaultInferenceParams;

enum class OutputLayout { LC, CL };

/**
 * Interface for running pre-processing for FacialLandmarks.
 */
class CVCORE_API FacialLandmarksPreProcessor {
public:
    /**
     * Removing the default constructor for FacialLandmarksPreProcessor.
     */
    FacialLandmarksPreProcessor() = delete;

    /*
     * Constructor for FacialLandmarksPreProcessor.
     * @param preProcessorParams Image preprocessing parameters.
     * @param modelInputParams Model input parameters.
     */
    FacialLandmarksPreProcessor(const ImagePreProcessingParams &preProcessorParams,
                                const ModelInputParams &modelInputParams);

    /**
     * Destructor for FacialLandmarksPreProcessor.
     */
    ~FacialLandmarksPreProcessor();

    /*
     * Running preprocessing for a batch of images.
     * @param preProcessedImagesBatch output images after preprocessing.
     * @param facesBBoxes faces bboxes to crop the faces region.
     * @param facesBBoxesScales the scale factors by which to scale facesBBoxes.
     * @param inputImagesBatch the input images to be preprocessed.
     * @param stream Cuda stream.
     */
    void execute(Tensor<NHWC, C1, F32> &preProcessedImagesBatch,
                 Array<BBox> &facesBBoxes,
                 const Array<float> &facesBBoxesScales,
                 const Tensor<NHWC, C3, U8> &inputImagesBatch,
                 cudaStream_t stream = 0);

private:
    struct FacialLandmarksPreProcessorImpl;
    std::unique_ptr<FacialLandmarksPreProcessorImpl> m_pImpl;
};

/**
 * Interface for loading and running FacialLandmarks.
 */
class CVCORE_API FacialLandmarks {
public:
    /**
     * The total number of Facial Landmarks.
     */
    static constexpr uint32_t MAX_NUM_FACIAL_LANDMARKS = 200;

    /**
     * Removing the default constructor for FacialLandmarks.
     */
    FacialLandmarks() = delete;

    /*
     * Constructor for FacialLandmarks.
     * @param preProcessorParams Image preprocessing parameters.
     * @param modelInputParams Model input parameters.
     * @param inferenceParams Inference parameters for the model.
     * @param numLandmarks Number of output landmarks
     * @param outputLayout whether the output landmarks are in order of LC...
     */
    FacialLandmarks(const ImagePreProcessingParams &preProcessorParams,
                    const ModelInputParams &modelInputParams,
                    const ModelInferenceParams &inferenceParams,
                    size_t numLandmarks = 80,
                    OutputLayout outputLayout = OutputLayout::LC);

    /**
     * Destructor for FacialLandmarks.
     */
    ~FacialLandmarks();

    /*
     * Running FacialLandmarks for a given image.
     * @param facialKeypointsCoordinates output facial keypoints for a given face.
     * @param facesBBoxes faces bboxes to crop the faces.
     * @param facesBBoxesScales the scale factor by wich to scale facesBBoxes.
     * @param inputImagesBatch the input images to be preprocessed
     * @param stream Cuda stream
     */
    void execute(Array<ArrayN<Vector2f, FacialLandmarks::MAX_NUM_FACIAL_LANDMARKS>>
                     &facialKeypointsCoordinates,
                 Array<BBox> &facesBBoxes,
                 const Array<float> &facesBBoxesScales,
                 const Tensor<NHWC, C3, U8> &inputImagesBatch,
                 cudaStream_t stream = 0);

private:
    struct FacialLandmarksImpl;
    std::unique_ptr<FacialLandmarksImpl> m_pImpl;
};

/**
 * Interface for running post-processing for FacialLandmarks.
 */
class CVCORE_API FacialLandmarksPostProcessor {
public:
    /**
     * Removing the default constructor for FacialLandmarksPostProcessor.
     */
    FacialLandmarksPostProcessor() = delete;

    /*
     * Constructor for FacialLandmarksPostProcessor.
     * @param modelInputParams Model input parameters.
     * @param numLandmarks Number of output landmarks.
     * @param outputLayout whether the output landmarks are in order of LC...
     */
    FacialLandmarksPostProcessor(const ModelInputParams &modelInputParams,
                                 size_t numLandmarks = 80,
                                 OutputLayout outputLayout = OutputLayout::LC);

    /**
     * Destructor for FacialLandmarksPostProcessor.
     */
    ~FacialLandmarksPostProcessor();

    /**
     * Allocate staging CPU buffers (used when inputs are GPU Tensors).
     */
    void allocateStagingBuffers();

    /*
     * Running postprocessing for a given image.
     * @param facialKeypointsCoordinates output facial keypoints for each image in the batch.
     * @param coordRaw the candidate facial keypoints for a given face.
     * @param facesBBoxes squarified faces bboxes.
     * @param stream Cuda stream
     */
    void execute(Array<ArrayN<Vector2f, FacialLandmarks::MAX_NUM_FACIAL_LANDMARKS>>
                     &facialKeypointsCoordinates,
                 const Tensor<CL, CX, F32> &coordRaw,
                 const Array<BBox> &facesBBoxes,
                 cudaStream_t stream = 0);

    /*
     * Map 126 landmarks Array to 68 landmarks model Array
     * @param outputLandmarks output array of 68 landmarks.
     * @param inputLandmarks input array of 126 landmarks.
     */
    static void map126LandmarksTo68(Array<Vector2f> &outputLandmarks,
                                    const Array<Vector2f> &inputLandmarks);

    /*
     * Map 126 landmarks ArrayN to 68 landmarks model Array
     * @param outputLandmarks output array of 68 landmarks.
     * @param inputLandmarks input array of 126 landmarks.
     */
    static void map126LandmarksTo68(
        Array<Vector2f> &outputLandmarks,
        const ArrayN<Vector2f, FacialLandmarks::MAX_NUM_FACIAL_LANDMARKS> &inputLandmarks);

    /*
     * Map 126 landmarks Array to 68 landmarks model ArrayN
     * @param outputLandmarks output array of 68 landmarks.
     * @param inputLandmarks input array of 126 landmarks.
     */
    static void map126LandmarksTo68(ArrayN<Vector2f, 68> &outputLandmarks,
                                    const Array<Vector2f> &inputLandmarks);

    /*
     * Map 126 landmarks ArrayN to 68 landmarks model ArrayN
     * @param outputLandmarks output array of 68 landmarks.
     * @param inputLandmarks input array of 126 landmarks.
     */
    static void map126LandmarksTo68(
        ArrayN<Vector2f, 68> &outputLandmarks,
        const ArrayN<Vector2f, FacialLandmarks::MAX_NUM_FACIAL_LANDMARKS> &inputLandmarks);

    /*
     * Map 126 batch landmarks Array to 68 batch landmarks model Array
     * @param outputLandmarks output array of 68 landmarks.
     * @param inputLandmarks input array of 126 landmarks.
     */
    static void map126LandmarksTo68(
        Array<ArrayN<Vector2f, 68>> &outputLandmarks,
        const Array<ArrayN<Vector2f, FacialLandmarks::MAX_NUM_FACIAL_LANDMARKS>> &inputLandmarks);

private:
    struct FacialLandmarksPostProcessorImpl;
    std::unique_ptr<FacialLandmarksPostProcessorImpl> m_pImpl;
};

} // namespace faciallandmarks
} // namespace cvcore

#endif
