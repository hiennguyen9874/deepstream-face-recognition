/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _DS3D_COMMON_COMMON__H
#define _DS3D_COMMON_COMMON__H

#include <ds3d/common/defines.h>
#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>

namespace ds3d {

enum class ErrCode : int {
    kGood = 0,
    kByPass = 1,
    KEndOfStream = 2,

    kLoadLib = -1,
    kMem = -2,
    kParam = -3,
    kNotFound = -4,
    kTimeOut = -5,
    kTypeId = -6,
    kNvDsMeta = -7,
    kUnsupported = -8,
    kConfig = -9,
    kRealSense = -10,
    kNullPtr = -11,
    kOutOfRange = -12,
    kGst = -13,
    kState = -14,
    kGL = -15,
    kLockWakeup = -16,
    kUnknown = -255,
};

using TIdType = uint64_t;

// structure TimeStamp
static constexpr const char *kTimeStamp = DS3D_KEY_NAME("Timestamp");
// get from Frame2DGuard
static constexpr const char *kColorFrame = DS3D_KEY_NAME("ColorFrame");
// get from Frame2DGuard
static constexpr const char *kDepthFrame = DS3D_KEY_NAME("DepthFrame");
// structure DepthScale
static constexpr const char *kDepthScaleUnit = DS3D_KEY_NAME("DepthScaleUnit");
// structure IntrinsicsParam
static constexpr const char *kDepthIntrinsics = DS3D_KEY_NAME("DepthIntrinsics");
// structure IntrinsicsParam
static constexpr const char *kColorIntrinsics = DS3D_KEY_NAME("ColorIntrinsics");
// structure ExtrinsicsParam
static constexpr const char *kDepth2ColorExtrinsics = DS3D_KEY_NAME("Depth2ColorExtrinsics");
// structure bool
static constexpr const char *kColorDepthAligned = DS3D_KEY_NAME("ColorDepthAligned");

// structure bool
static constexpr const char *kEOS = DS3D_KEY_NAME("EndOfStream");
// get from FrameGuard
static constexpr const char *kPointXYZ = DS3D_KEY_NAME("PointXYZ");
// get from FrameGuard
static constexpr const char *kPointCoordUV = DS3D_KEY_NAME("PointColorCoord");

// default caps for input and ouptut
static constexpr const char *kDefaultDs3dCaps = "ds3d/datamap";

class Exception : public std::exception {
    std::string _msg;
    ErrCode _code = ErrCode::kGood;

public:
    Exception(ErrCode code, const std::string &s = "") : _msg(s), _code(code) {}
    ErrCode code() const { return _code; }
    const char *what() const noexcept override { return _msg.c_str(); }
};

} // namespace ds3d

#endif // _DS3D_COMMON_COMMON__H