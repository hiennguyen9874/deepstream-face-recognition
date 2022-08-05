/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CVCORE_IMAGE_H
#define CVCORE_IMAGE_H

#include <cassert>
#include <functional>
#include <tuple>

#include "Tensor.h"

namespace cvcore {

/**
 * An enum.
 * Enum type for image type.
 */
enum ImageType {
    Y_U8,            /**< 8-bit gray. */
    Y_U16,           /**< 16-bit gray. */
    Y_F16,           /**< half normalized gray. */
    Y_F32,           /**< float normalized gray. */
    RGB_U8,          /**< 8-bit RGB. */
    RGB_U16,         /**< 16-bit RGB. */
    RGB_F16,         /**< half RGB. */
    RGB_F32,         /**< float RGB. */
    BGR_U8,          /**< 8-bit BGR. */
    BGR_U16,         /**< 16-bit BGR. */
    BGR_F16,         /**< half BGR. */
    BGR_F32,         /**< float BGR. */
    RGBA_U8,         /**< 8-bit RGBA. */
    RGBA_U16,        /**< 16-bit RGBA. */
    RGBA_F16,        /**< half RGBA. */
    RGBA_F32,        /**< float RGBA. */
    PLANAR_RGB_U8,   /**< 8-bit planar RGB. */
    PLANAR_RGB_U16,  /**< 16-bit planar RGB. */
    PLANAR_RGB_F16,  /**< half planar RGB. */
    PLANAR_RGB_F32,  /**< float planar RGB. */
    PLANAR_BGR_U8,   /**< 8-bit planar BGR. */
    PLANAR_BGR_U16,  /**< 16-bit planar BGR. */
    PLANAR_BGR_F16,  /**< half planar BGR. */
    PLANAR_BGR_F32,  /**< float planar BGR. */
    PLANAR_RGBA_U8,  /**< 8-bit planar RGBA. */
    PLANAR_RGBA_U16, /**< 16-bit planar RGBA. */
    PLANAR_RGBA_F16, /**< half planar RGBA. */
    PLANAR_RGBA_F32, /**< float planar RGBA. */
    NV12,            /**< 8-bit planar Y + interleaved and subsampled (1/4 Y samples) UV. */
};

/**
 * Struct type for image preprocessing params
 */
struct ImagePreProcessingParams {
    ImageType imgType;      /**< Input Image Type. */
    float pixelMean[3];     /**< Image Mean value offset for R,G,B channels. Default is 0.0f */
    float normalization[3]; /**< Scale or normalization values for  R,G,B channels. Default
                               is 1.0/255.0f */
    float stdDev[3];        /**< Standard deviation values for  R,G,B channels. Default is 1.0f */
};

/**
 * Image traits that map ImageType to TensorLayout, ChannelCount and ChannelType.
 */
template <ImageType T, size_t N>
struct ImageTraits;

template <>
struct ImageTraits<Y_U8, 3> {
    static constexpr TensorLayout TL = TensorLayout::HWC;
    static constexpr ChannelCount CC = ChannelCount::C1;
    static constexpr ChannelType CT = ChannelType::U8;
};

template <>
struct ImageTraits<Y_U8, 4> {
    static constexpr TensorLayout TL = TensorLayout::NHWC;
    static constexpr ChannelCount CC = ChannelCount::C1;
    static constexpr ChannelType CT = ChannelType::U8;
};

template <>
struct ImageTraits<Y_U16, 3> {
    static constexpr TensorLayout TL = TensorLayout::HWC;
    static constexpr ChannelCount CC = ChannelCount::C1;
    static constexpr ChannelType CT = ChannelType::U16;
};

template <>
struct ImageTraits<Y_U16, 4> {
    static constexpr TensorLayout TL = TensorLayout::NHWC;
    static constexpr ChannelCount CC = ChannelCount::C1;
    static constexpr ChannelType CT = ChannelType::U16;
};

template <>
struct ImageTraits<Y_F32, 3> {
    static constexpr TensorLayout TL = TensorLayout::HWC;
    static constexpr ChannelCount CC = ChannelCount::C1;
    static constexpr ChannelType CT = ChannelType::F32;
};

template <>
struct ImageTraits<Y_F32, 4> {
    static constexpr TensorLayout TL = TensorLayout::NHWC;
    static constexpr ChannelCount CC = ChannelCount::C1;
    static constexpr ChannelType CT = ChannelType::F32;
};

template <>
struct ImageTraits<RGB_U8, 3> {
    static constexpr TensorLayout TL = TensorLayout::HWC;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U8;
};

template <>
struct ImageTraits<RGB_U8, 4> {
    static constexpr TensorLayout TL = TensorLayout::NHWC;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U8;
};

template <>
struct ImageTraits<RGB_U16, 3> {
    static constexpr TensorLayout TL = TensorLayout::HWC;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U16;
};

template <>
struct ImageTraits<RGB_U16, 4> {
    static constexpr TensorLayout TL = TensorLayout::NHWC;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U16;
};

template <>
struct ImageTraits<RGB_F32, 3> {
    static constexpr TensorLayout TL = TensorLayout::HWC;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::F32;
};

template <>
struct ImageTraits<RGB_F32, 4> {
    static constexpr TensorLayout TL = TensorLayout::NHWC;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::F32;
};

template <>
struct ImageTraits<BGR_U8, 3> {
    static constexpr TensorLayout TL = TensorLayout::HWC;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U8;
};

template <>
struct ImageTraits<BGR_U8, 4> {
    static constexpr TensorLayout TL = TensorLayout::NHWC;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U8;
};

template <>
struct ImageTraits<BGR_U16, 3> {
    static constexpr TensorLayout TL = TensorLayout::HWC;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U16;
};

template <>
struct ImageTraits<BGR_U16, 4> {
    static constexpr TensorLayout TL = TensorLayout::NHWC;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U16;
};

template <>
struct ImageTraits<BGR_F32, 3> {
    static constexpr TensorLayout TL = TensorLayout::HWC;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::F32;
};

template <>
struct ImageTraits<BGR_F32, 4> {
    static constexpr TensorLayout TL = TensorLayout::NHWC;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::F32;
};

template <>
struct ImageTraits<PLANAR_RGB_U8, 3> {
    static constexpr TensorLayout TL = TensorLayout::CHW;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U8;
};

template <>
struct ImageTraits<PLANAR_RGB_U8, 4> {
    static constexpr TensorLayout TL = TensorLayout::NCHW;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U8;
};

template <>
struct ImageTraits<PLANAR_RGB_U16, 3> {
    static constexpr TensorLayout TL = TensorLayout::CHW;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U16;
};

template <>
struct ImageTraits<PLANAR_RGB_U16, 4> {
    static constexpr TensorLayout TL = TensorLayout::NCHW;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U16;
};

template <>
struct ImageTraits<PLANAR_RGB_F32, 3> {
    static constexpr TensorLayout TL = TensorLayout::CHW;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::F32;
};

template <>
struct ImageTraits<PLANAR_RGB_F32, 4> {
    static constexpr TensorLayout TL = TensorLayout::NCHW;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::F32;
};

template <>
struct ImageTraits<PLANAR_BGR_U8, 3> {
    static constexpr TensorLayout TL = TensorLayout::CHW;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U8;
};

template <>
struct ImageTraits<PLANAR_BGR_U8, 4> {
    static constexpr TensorLayout TL = TensorLayout::NCHW;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U8;
};

template <>
struct ImageTraits<PLANAR_BGR_U16, 3> {
    static constexpr TensorLayout TL = TensorLayout::CHW;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U16;
};

template <>
struct ImageTraits<PLANAR_BGR_U16, 4> {
    static constexpr TensorLayout TL = TensorLayout::NCHW;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::U16;
};

template <>
struct ImageTraits<PLANAR_BGR_F32, 3> {
    static constexpr TensorLayout TL = TensorLayout::CHW;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::F32;
};

template <>
struct ImageTraits<PLANAR_BGR_F32, 4> {
    static constexpr TensorLayout TL = TensorLayout::NCHW;
    static constexpr ChannelCount CC = ChannelCount::C3;
    static constexpr ChannelType CT = ChannelType::F32;
};

/**
 * Get the bytes of each element for a specific ImageType.
 */
inline size_t GetImageElementSize(const ImageType type)
{
    size_t imageElementSize;

    switch (type) {
    case ImageType::Y_U8:
    case ImageType::RGB_U8:
    case ImageType::BGR_U8:
    case ImageType::RGBA_U8:
    case ImageType::PLANAR_RGB_U8:
    case ImageType::PLANAR_BGR_U8:
    case ImageType::PLANAR_RGBA_U8: {
        imageElementSize = 1;
        break;
    }
    case ImageType::Y_U16:
    case ImageType::RGB_U16:
    case ImageType::BGR_U16:
    case ImageType::RGBA_U16:
    case ImageType::PLANAR_RGB_U16:
    case ImageType::PLANAR_BGR_U16:
    case ImageType::PLANAR_RGBA_U16:
    case ImageType::Y_F16:
    case ImageType::RGB_F16:
    case ImageType::BGR_F16:
    case ImageType::RGBA_F16:
    case ImageType::PLANAR_RGB_F16:
    case ImageType::PLANAR_BGR_F16:
    case ImageType::PLANAR_RGBA_F16: {
        imageElementSize = 2;
        break;
    }
    case ImageType::Y_F32:
    case ImageType::RGB_F32:
    case ImageType::BGR_F32:
    case ImageType::RGBA_F32:
    case ImageType::PLANAR_RGB_F32:
    case ImageType::PLANAR_BGR_F32:
    case ImageType::PLANAR_RGBA_F32: {
        imageElementSize = 4;
        break;
    }
    default: {
        imageElementSize = 0;
    }
    }

    return imageElementSize;
}

/**
 * Get the number of channels for a specific ImageType.
 */
inline size_t GetImageChannelCount(const ImageType type)
{
    size_t imageChannelCount;

    switch (type) {
    case ImageType::Y_U8:
    case ImageType::Y_U16:
    case ImageType::Y_F16:
    case ImageType::Y_F32: {
        imageChannelCount = 1;
        break;
    }
    case ImageType::RGB_U8:
    case ImageType::RGB_U16:
    case ImageType::RGB_F16:
    case ImageType::RGB_F32:
    case ImageType::BGR_U8:
    case ImageType::BGR_U16:
    case ImageType::BGR_F16:
    case ImageType::BGR_F32:
    case ImageType::PLANAR_RGB_U8:
    case ImageType::PLANAR_RGB_U16:
    case ImageType::PLANAR_RGB_F16:
    case ImageType::PLANAR_RGB_F32:
    case ImageType::PLANAR_BGR_U8:
    case ImageType::PLANAR_BGR_U16:
    case ImageType::PLANAR_BGR_F16:
    case ImageType::PLANAR_BGR_F32: {
        imageChannelCount = 3;
        break;
    }
    case ImageType::RGBA_U8:
    case ImageType::RGBA_U16:
    case ImageType::RGBA_F16:
    case ImageType::RGBA_F32:
    case ImageType::PLANAR_RGBA_U8:
    case ImageType::PLANAR_RGBA_U16:
    case ImageType::PLANAR_RGBA_F16:
    case ImageType::PLANAR_RGBA_F32: {
        imageChannelCount = 4;
        break;
    }
    default: {
        imageChannelCount = 0;
    }
    }

    return imageChannelCount;
};

template <ImageType T>
class Image {
};

template <>
class Image<ImageType::Y_U8> : public Tensor<HWC, C1, U8> {
    using Tensor<HWC, C1, U8>::Tensor;
};

template <>
class Image<ImageType::Y_U16> : public Tensor<HWC, C1, U16> {
    using Tensor<HWC, C1, U16>::Tensor;
};

template <>
class Image<ImageType::Y_F16> : public Tensor<HWC, C1, F16> {
    using Tensor<HWC, C1, F16>::Tensor;
};

template <>
class Image<ImageType::Y_F32> : public Tensor<HWC, C1, F32> {
    using Tensor<HWC, C1, F32>::Tensor;
};

template <>
class Image<ImageType::RGB_U8> : public Tensor<HWC, C3, U8> {
    using Tensor<HWC, C3, U8>::Tensor;
};

template <>
class Image<ImageType::RGB_U16> : public Tensor<HWC, C3, U16> {
    using Tensor<HWC, C3, U16>::Tensor;
};

template <>
class Image<ImageType::RGB_F16> : public Tensor<HWC, C3, F16> {
    using Tensor<HWC, C3, F16>::Tensor;
};

template <>
class Image<ImageType::RGB_F32> : public Tensor<HWC, C3, F32> {
    using Tensor<HWC, C3, F32>::Tensor;
};

template <>
class Image<ImageType::BGR_U8> : public Tensor<HWC, C3, U8> {
    using Tensor<HWC, C3, U8>::Tensor;
};

template <>
class Image<ImageType::BGR_U16> : public Tensor<HWC, C3, U16> {
    using Tensor<HWC, C3, U16>::Tensor;
};

template <>
class Image<ImageType::BGR_F16> : public Tensor<HWC, C3, F16> {
    using Tensor<HWC, C3, F16>::Tensor;
};

template <>
class Image<ImageType::BGR_F32> : public Tensor<HWC, C3, F32> {
    using Tensor<HWC, C3, F32>::Tensor;
};

template <>
class Image<ImageType::RGBA_U8> : public Tensor<HWC, C4, U8> {
    using Tensor<HWC, C4, U8>::Tensor;
};

template <>
class Image<ImageType::RGBA_U16> : public Tensor<HWC, C4, U16> {
    using Tensor<HWC, C4, U16>::Tensor;
};

template <>
class Image<ImageType::RGBA_F16> : public Tensor<HWC, C4, F16> {
    using Tensor<HWC, C4, F16>::Tensor;
};

template <>
class Image<ImageType::RGBA_F32> : public Tensor<HWC, C4, F32> {
    using Tensor<HWC, C4, F32>::Tensor;
};

template <>
class Image<ImageType::PLANAR_RGB_U8> : public Tensor<CHW, C3, U8> {
    using Tensor<CHW, C3, U8>::Tensor;
};

template <>
class Image<ImageType::PLANAR_RGB_U16> : public Tensor<CHW, C3, U16> {
    using Tensor<CHW, C3, U16>::Tensor;
};

template <>
class Image<ImageType::PLANAR_RGB_F16> : public Tensor<CHW, C3, F16> {
    using Tensor<CHW, C3, F16>::Tensor;
};

template <>
class Image<ImageType::PLANAR_RGB_F32> : public Tensor<CHW, C3, F32> {
    using Tensor<CHW, C3, F32>::Tensor;
};

template <>
class Image<ImageType::PLANAR_RGBA_U8> : public Tensor<CHW, C4, U8> {
    using Tensor<CHW, C4, U8>::Tensor;
};

template <>
class Image<ImageType::PLANAR_RGBA_U16> : public Tensor<CHW, C4, U16> {
    using Tensor<CHW, C4, U16>::Tensor;
};

template <>
class Image<ImageType::PLANAR_RGBA_F16> : public Tensor<CHW, C4, F16> {
    using Tensor<CHW, C4, F16>::Tensor;
};

template <>
class Image<ImageType::PLANAR_RGBA_F32> : public Tensor<CHW, C4, F32> {
    using Tensor<CHW, C4, F32>::Tensor;
};

template <>
class Image<ImageType::NV12> {
public:
    Image(std::size_t width, std::size_t height, bool isCPU = true)
        : m_data(std::make_tuple(Y(width, height, isCPU), UV(width / 2, height / 2, isCPU)))
    {
        assert(width % 2 == 0 && height % 2 == 0);
    }

    Image(std::size_t width,
          std::size_t height,
          std::uint8_t *dataPtr1,
          std::uint8_t *dataPtr2,
          bool isCPU = true)
        : m_data(std::make_tuple(Y(width, height, dataPtr1, isCPU),
                                 UV(width / 2, height / 2, dataPtr2, isCPU)))
    {
        assert(width % 2 == 0 && height % 2 == 0);
    }

    std::size_t getLumaWidth() const { return std::get<0>(m_data).getWidth(); }

    std::size_t getLumaHeight() const { return std::get<0>(m_data).getHeight(); }

    std::size_t getChromaWidth() const { return std::get<1>(m_data).getWidth(); }

    std::size_t getChromaHeight() const { return std::get<1>(m_data).getHeight(); }

    std::uint8_t *getLumaData() { return std::get<0>(m_data).getData(); }

    std::uint8_t *getChromaData() { return std::get<1>(m_data).getData(); }

private:
    using Y = Tensor<HWC, C1, U8>;
    using UV = Tensor<HWC, C2, U8>;

    std::tuple<Y, UV> m_data;
};

} // namespace cvcore

#endif // CVCORE_IMAGE_H
