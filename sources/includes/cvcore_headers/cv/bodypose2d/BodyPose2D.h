/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NV_BODYPOSE_2D_H_
#define NV_BODYPOSE_2D_H_

#include <cuda_runtime.h>
#include <cv/core/Array.h>
#include <cv/core/BBox.h>
#include <cv/core/Core.h>
#include <cv/core/Image.h>
#include <cv/core/MathTypes.h>
#include <cv/core/Model.h>
#include <cv/core/Tensor.h>

#include <map>
#include <memory>

namespace cvcore {
namespace bodypose2d {

/**
 * Data structure for storing body part location.
 */
struct BodyPart {
    int part_idx;       /**< part index. */
    Vector2i loc;       /**< x, y pixel location. */
    float score = 0.0f; /**< part score. */
};

/**
 * Data structure for storing human info.
 */
struct Human {
    std::map<int, BodyPart> body_parts; /**< body parts map. */
    BBox boundingBox;                   /**< bounding box for person. */
    float score = 0.0f;                 /**< person score. */
};

/**
 * Data structure to describe the post processing for BodyPose2D
 */
struct BodyPose2DPostProcessorParams {
    size_t numJoints;            /**< Number of joints. */
    size_t nmsWindowSize;        /**< window size for NMS operation. */
    size_t featUpsamplingFactor; /**< feature upsampleing factor. */
    float threshHeat;            /**< model threshold for heatmap. */
    float threshVectorScore;     /**< model threshold for vector score. */
    int threshVectorCnt1;        /**< model threshold for vector count. */
    int threshPartCnt;           /**< model threshold for part count. */
    float threshHumanScore;      /**< model threshold for human score. */
    std::vector<std::pair<uint32_t, uint32_t>>
        jointEdges; /** Skeleton edge mapping using zero indexed joint indices. Example: If joint
                       index 0 is nose and index 4 is left ear, {0, 4} represents an edge between
                       nose tip and left ear, Joint edges need to be provided in the order of paf
                       map order. i.e Edge 0 corresponds to channel 0, 1 of paf Map, Edge index 1
                       corresponds to channel 2, 3 of paf map*/
};

/**
 * Enum listing the hand used for gestures
 */
enum class HandType : uint32_t { LEFT_HAND = 0, RIGHT_HAND };

/**
 * Data structure to describe the parameters tuned for bounding box detection from pose.
 */
struct HandBoundingBoxParams {
    HandType handType;                  /**< Left or Right hand. */
    bool enableHandBoxDynamicScaling;   /**< Enable using hand scale input */
    float defaultHandBoxScalingFactor;  /** Default scaling factor for bounding box*/
    float handBoxScalingFactorMinRange; /**< Minimum scaling factor */
    float handBoxScalingFactorMaxRange; /**< Maximum scaling factor */
    Vector2d
        handBoxScalingFactorTrainDims; /** Training factor used to determine the scaling factor. */
    Vector2f handBoxScalingFactorNeckHipCoeff; /**< Scale ratio between Neck and Hip*/
    float handBoxNoseNeckHipRatio;             /** Scale factor between neck and nose*/
    /* Wrist and elbow indices are necessary for computing the hand bounding box*/
    uint32_t wristIndex; /** Left or right wrist index of the pose provided*/
    uint32_t elbowIndex; /** Left or right elbow index of the pose provided*/
    /* Hip, nose and neck joint indices are optional.
     * If these indices are not available, the index can be set to any
     * invalid index value i.e index > numJoints */
    uint32_t leftHipIndex;  /** Left hip index of the pose provided*/
    uint32_t rightHipIndex; /** Right hip index of the pose provided*/
    uint32_t noseIndex;     /** Nose index of the pose provided*/
    uint32_t neckIndex;     /** Neck index of the pose provided*/
};

/**
 *  Default parameters for computing hand bounding box from pose.
 */
CVCORE_API extern const HandBoundingBoxParams defaultHandParams;

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
CVCORE_API extern const BodyPose2DPostProcessorParams defaultPostProcessorParams;

/**
 * Interface for running pre-processing on bodypose2d network.
 */
class CVCORE_API BodyPose2DPreProcessor {
public:
    /**
     * Default constructor is deleted
     */
    BodyPose2DPreProcessor() = delete;

    /**
     * Constructor of BodyPose2DPreProcessor.
     * @param preProcessorParams image pre-processing parameters.
     * @param modelInputParams model paramters for network.
     */
    BodyPose2DPreProcessor(const ImagePreProcessingParams &preProcessorParams,
                           const ModelInputParams &modelInputParams);

    /**
     * Destructor of BodyPose2DPreProcessor.
     */
    ~BodyPose2DPreProcessor();

    /**
     * Main interface to run pre-processing for interleaved output.
     * @param output output tensor.
     * @param input input tensor.
     * @param stream cuda stream.
     */
    void execute(Tensor<NHWC, C3, F32> &output,
                 const Tensor<NHWC, C3, U8> &input,
                 cudaStream_t stream = 0);

    /**
     * Main interface to run pre-processing for planar output.
     * @param output output tensor.
     * @param input input tensor.
     * @param stream cuda stream.
     */
    void execute(Tensor<NCHW, C3, F32> &output,
                 const Tensor<NHWC, C3, U8> &input,
                 cudaStream_t stream = 0);

private:
    /**
     * Implementation of Pose2DPostProcessor.
     */
    struct BodyPose2DPreProcessorImpl;
    std::unique_ptr<BodyPose2DPreProcessorImpl> m_pImpl;
};

/**
 * Interface for loading and running bodypose2d network.
 */
class CVCORE_API BodyPose2D {
public:
    /**
     * Maximum detected human allowed.
     */
    static constexpr int MAX_HUMAN_COUNT = 200;

    /**
     * Default constructor is deleted
     */
    BodyPose2D() = delete;

    /**
     * Constructor of BodyPose2D.
     * @param preProcessorParams custom pre-processor params.
     * @param modelInputParams custom model input params for the network.
     * @param inferenceParams custom inference params for the network.
     * @param postProcessorParams custom post-processor params.
     */
    BodyPose2D(const ImagePreProcessingParams &preProcessorParams,
               const ModelInputParams &modelInputParams,
               const ModelInferenceParams &inferenceParams,
               const BodyPose2DPostProcessorParams &postProcessorParams);

    /**
     * Destructor of BodyPose2D.
     */
    ~BodyPose2D();

    /**
     * Main interface to run inference for batch input.
     * @param output output bodypose detection batch results.
     * @param input input batch tensor as interleaved BGR image.
     * @param stream cuda stream.
     */
    void execute(Array<ArrayN<Human, MAX_HUMAN_COUNT>> &output,
                 const Tensor<NHWC, C3, U8> &input,
                 cudaStream_t stream = 0);

private:
    /**
     * Implementation of bodypose2d.
     */
    struct BodyPose2DImpl;
    std::unique_ptr<BodyPose2DImpl> m_pImpl;
};

/**
 * Interface for running post-processing on bodypose2d network.
 */
class CVCORE_API BodyPose2DPostProcessor {
public:
    /**
     * Default constructor is deleted
     */
    BodyPose2DPostProcessor() = delete;

    /**
     * Constructor of BodyPose2DPostProcessor.
     * @param params bodypose2d post-processing parameters.
     * @param modelInputParams model paramters for network.
     */
    BodyPose2DPostProcessor(const BodyPose2DPostProcessorParams &params,
                            const ModelInputParams &modelInputParams);

    /**
     * Destructor of BodyPose2DPostProcessor.
     */
    ~BodyPose2DPostProcessor();

    /**
     * Allocate staging CPU buffers (used when inputs are GPU Tensors).
     */
    void allocateStagingBuffers();

    /**
     * Main interface to run post-processing for batch input.
     * @param output output bodypose2d detection results.
     * @param pafMap input part-affinity-field layer.
     * @param heatMap input heatmap layer.
     * @param imageWidth input image width.
     * @param imageHeight input image height.
     */
    void execute(Array<ArrayN<Human, BodyPose2D::MAX_HUMAN_COUNT>> &output,
                 const Tensor<NCHW, CX, F32> &pafMap,
                 const Tensor<NCHW, CX, F32> &heatMap,
                 int imageWidth,
                 int imageHeight,
                 cudaStream_t stream = 0);

private:
    /**
     * Implementation of Pose2DPostProcessor.
     */
    struct BodyPose2DPostProcessorImpl;
    std::unique_ptr<BodyPose2DPostProcessorImpl> m_pImpl;
};

/**
 * Class to compute the bounding box of hand from pose.
 */
class CVCORE_API HandBoundingBoxGenerator {
public:
    static constexpr size_t MOVING_AVG_WINDOW_SIZE = 100;
    /**
     * Default constructor is deleted
     */
    HandBoundingBoxGenerator() = delete;
    /**
     * Constructor of HandBoundingBoxGenerator
     * @param parameters configured for hand detection.
     */
    HandBoundingBoxGenerator(const HandBoundingBoxParams &params);
    /**
     * Get bounding box values from input pose.
     * @param 2D pose computed
     * @param Image Width
     * @param Image Height
     * @param Output Bounding box computed
     * @return Validity of the bounding box computed.
     */

    bool execute(BBox &box, const Human &pose, size_t imageWidth, size_t imageHeight);

    /**
     * Destructor of HandBoundingBoxGenerator
     */
    ~HandBoundingBoxGenerator();

private:
    /**
     * Implmentation of HandBoundingBoxGenerator
     */
    struct HandBoundingBoxGeneratorImpl;
    std::unique_ptr<HandBoundingBoxGeneratorImpl> m_pImpl;
};

} // namespace bodypose2d
} // namespace cvcore
#endif
