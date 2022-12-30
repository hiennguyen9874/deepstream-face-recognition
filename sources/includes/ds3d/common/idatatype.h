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

#ifndef _DS3D_COMMON_IDATATYPE__H
#define _DS3D_COMMON_IDATATYPE__H

#include <ds3d/common/common.h>
#include <ds3d/common/typeid.h>

namespace ds3d {

template <typename T>
union __xyzw {
    T data[4];
    struct {
        T x, y, z, w;
    };
};

template <typename T>
union __xyz {
    T data[3];
    struct {
        T x, y, z;
    };
};

template <typename T>
union __xy {
    T data[2];
    struct {
        T x, y;
    };
};

template <typename T>
union __x {
    T data[1];
    struct {
        T x;
    };
};

template <typename T>
using vec4 = __xyzw<T>;
template <typename T>
using vec3 = __xyz<T>;
template <typename T>
using vec2 = __xy<T>;
template <typename T>
using vec1 = __x<T>;

using vec4f = vec4<float>;
using vec3f = vec3<float>;
using vec2f = vec2<float>;
using vec1f = vec1<float>;

using vec4b = vec4<uint8_t>;
using vec3b = vec3<uint8_t>;
using vec2b = vec2<uint8_t>;
using vec1b = vec1<uint8_t>;

// keep same order with InferDataType
enum class DataType : int {
    kFp32 = 0,  // 0
    kFp16 = 1,  // 1
    kInt8 = 2,  // 2
    kInt32 = 3, // 3
    kInt16 = 7,
    kUint8,
    kUint16,
    kUint32,
    kDouble,
};

enum class FrameType : int {
    kUnknown = 0,
    kDepth = 1,
    kColorRGBA = 2,
    kColorRGB = 3,
    kPointXYZ = 32,
    kPointCoordUV = 33,
    kCustom = 255,
};

enum class MemType : int {
    kNone = 0,
    kGpuCuda = 1,
    kCpu = 2,
    kCpuPinned = 3,
};

struct TimeStamp { // unit in nanosecond
    uint64_t t0 = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    REGISTER_TYPE_ID(DS3D_TYPEID_TIMESTAMP)
};

constexpr const size_t kMaxShapeDims = 8;

struct Shape {
    uint32_t numDims = 0;
    int32_t d[kMaxShapeDims] = {0};
    REGISTER_TYPE_ID(DS3D_TYPEID_SHAPE)
};

struct DepthScale {
    double scaleUnit = 0.0f;
    void *reserved[3] = {nullptr};
    REGISTER_TYPE_ID(DS3D_TYPEID_DEPTH_SCALE)
};

struct IntrinsicsParam {
    uint32_t width = 0;
    uint32_t height = 0;
    float centerX = 0; // coordinate axis in horizontal
    float centerY = 0; // coordinate axis position in vertical
    float fx = 0;      // focal lenth in x
    float fy = 0;      // focal lenth in y
    REGISTER_TYPE_ID(DS3D_TYPEID_INTRINSIC_PARM)
};

struct ExtrinsicsParam {
    vec3f rotation[3]; // column-major order
    vec3f translation;
    REGISTER_TYPE_ID(DS3D_TYPEID_EXTRINSIC_PARM)
};

} // namespace ds3d

#endif // _DS3D_COMMON_IDATATYPE__H