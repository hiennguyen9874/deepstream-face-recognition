/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CVCORE_MEMORY_H
#define CVCORE_MEMORY_H

#include <cuda_runtime.h>

#include "Tensor.h"

namespace cvcore {

/**
 * Implementation of tensor copy.
 * @param dst destination TensorBase.
 * @param src source TensorBase.
 * @param stream cuda stream.
 */
void TensorBaseCopy(TensorBase &dst, const TensorBase &src, cudaStream_t stream = 0);

/**
 * Implementation of tensor copy for 2D pitch linear tensors.
 * @param dst destination TensorBase.
 * @param src source TensorBase.
 * @param dstPitch pitch of destination Tensor in bytes.
 * @param srcPitch pitch of source Tensor in bytes.
 * @param widthInBytes width in bytes.
 * @param height height of tensor.
 * @param stream cuda stream.
 */
void TensorBaseCopy2D(TensorBase &dst,
                      const TensorBase &src,
                      int dstPitch,
                      int srcPitch,
                      int widthInBytes,
                      int height,
                      cudaStream_t stream = 0);

/**
 * Memory copy function between two non HWC/NHWC Tensors.
 * @tparam TL TensorLayout type.
 * @tparam CC Channel Count.
 * @tparam CT ChannelType.
 * @param dst destination Tensor.
 * @param src source Tensor which copy from.
 * @param stream cuda stream.
 */
template <TensorLayout TL,
          ChannelCount CC,
          ChannelType CT,
          typename std::enable_if<TL != HWC && TL != NHWC>::type * = nullptr>
void Copy(Tensor<TL, CC, CT> &dst, const Tensor<TL, CC, CT> &src, cudaStream_t stream = 0)
{
    TensorBaseCopy(dst, src, stream);
}

/**
 * Memory copy function between two HWC Tensors.
 * @tparam TL TensorLayout type.
 * @tparam CC Channel Count.
 * @tparam CT ChannelType.
 * @param dst destination Tensor.
 * @param src source Tensor which copy from.
 * @param stream cuda stream.
 */
template <TensorLayout TL,
          ChannelCount CC,
          ChannelType CT,
          typename std::enable_if<TL == HWC>::type * = nullptr>
void Copy(Tensor<TL, CC, CT> &dst, const Tensor<TL, CC, CT> &src, cudaStream_t stream = 0)
{
    TensorBaseCopy2D(dst, src, dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                     src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                     dst.getWidth() * dst.getChannelCount() * GetChannelSize(CT), src.getHeight(),
                     stream);
}

/**
 * Memory copy function between two NHWC Tensors.
 * @tparam TL TensorLayout type.
 * @tparam CC Channel Count.
 * @tparam CT ChannelType.
 * @param dst destination Tensor.
 * @param src source Tensor which copy from.
 * @param stream cuda stream.
 */
template <TensorLayout TL,
          ChannelCount CC,
          ChannelType CT,
          typename std::enable_if<TL == NHWC>::type * = nullptr>
void Copy(Tensor<TL, CC, CT> &dst, const Tensor<TL, CC, CT> &src, cudaStream_t stream = 0)
{
    TensorBaseCopy2D(dst, src, dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                     src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                     dst.getWidth() * dst.getChannelCount() * GetChannelSize(CT),
                     src.getDepth() * src.getHeight(), stream);
}

} // namespace cvcore

#endif // CVCORE_MEMORY_H
