/*
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#include <cv/core/Array.h>
#include <cv/core/MathTypes.h>

#include <vector>

namespace cvcore {
namespace vision3d {

/**
 * A struct to describe the Pose hypothesis returned by RANSAC-based pose
 * estimation.
 * @param pose 3D pose estimated after refitting to all inliers.
 * @param score Score of the pose (prior to refitting).
 * @param  Indices of all inliers for this hypothesis.
 */
struct PoseHypothesis {
    Pose3d pose;
    double score;
    std::vector<uint32_t> inliers;
};

/**
 * A function to calculate the number of RANSAC rounds necessary to sample at
 * least a single uncontaminated sample set with a certain success rate given
 * the expected ratio of outliers.
 * @param successRate :  Required probability in the range (0,0.9999] of finding
 * the solution. As success_rate tends to 1.0, the number of experiments tends
 * to Inf.
 * @param outlierRatio : Maximum expected ratio of outliers in the open interval
 * [0,0.9].
 * @param sampleSize : The minimum number of samples necessary to fit the model
 * (at least 1).
 * @return  Returns the number of experiments (RANSAC iterations) from the
 * formula.
 */
uint32_t EvaluateRansacFormula(const float successRate,
                               const float outlierRatio,
                               const uint32_t sampleSize);

/**
 * A function to compute the pose of a camera give the camera instrinsics
 * and at least 6 2D-3D point correspondences without outliers.
 * Assumptions No lens distortions, you need to un-distort points in advance.
 * Using Right handed coordinate system.
 * @param points3 Array of 3d world coordinates
 * @param points2 Array of 2d projection points in pixels.
 * @param focal Focal length in horizontal axis and vertical axis in pixel
 * units.
 * @param principal Principal point / image center in x axis and y axis.
 * @return camera pose: rigid transformation that maps 3D points from the world
 * frame into the camera frame.
 */
Pose3d ComputeCameraPoseEpnp(const Array<Vector3d> &points3,
                             const Array<Vector2d> &points2,
                             const Vector2d focal,
                             const Vector2d principal);

/**
 * A function to compute the pose of a pinhole camera from at
 * least 6 2D-3D point correspondences with outliers using RANSAC.
 * @param points3 Array of 3d world coordinates
 * @param points2 Array of 2d projection points in pixels.
 * @param focal Focal length in horizontal axis and vertical axis in pixel
 * units.
 * @param principal Principal point / image center in x axis and y axis.
 * @param numExperiments Number of RANSAC iterations or experiments to run.
 * @param ransacThreshold  RANSAC threshold in terms of reprojection error in
 * pixels.
 * @param maxTopPoses Maximum number of pose hypotheses to return.
 * @param seed Integer seed for random number generation.
 * @return Returns the most top-K pose hypotheses in decreasing order of score.
 * Each pose hypothesis has min 6 inliers and is a rigid transformation mapping
 * 3D points from world to camera frame.
 * */
std::vector<PoseHypothesis> ComputeCameraPoseEpnpRansac(const Array<Vector3d> &points3,
                                                        const Array<Vector2d> &points2,
                                                        const Vector2d focal,
                                                        const Vector2d principal,
                                                        const uint32_t numExperiments,
                                                        const double ransacThreshold,
                                                        const uint32_t maxTopPoses,
                                                        const uint32_t seed);

/**
 * A function to compute the pose of a camera give the camera instrinsics iteratively.
 * Assumptions No lens distortions, you need to un-distort points in advance.
 * Using Right handed coordinate system.
 * @param points3 Array of 3d world coordinates
 * @param points2 Array of 2d projection points in pixels.
 * @param focal Focal length in horizontal axis and vertical axis in pixel
 * units.
 * @param principal Principal point / image center in x axis and y axis.
 * @param initialGuess Inital condition for the Iterative solver.
 * @param iterations Iterations of Iterative solver.
 * @return camera pose: rigid transformation that maps 3D points from the world
 * frame into the camera frame.
 */
Pose3d ComputeCameraPoseIterativePnp(const Array<Vector3d> &points3,
                                     const Array<Vector2d> &points2,
                                     const Vector2d focal,
                                     const Vector2d principal,
                                     const Pose3d &initialGuess,
                                     size_t iterations = 5);
} // namespace vision3d
} // namespace cvcore
