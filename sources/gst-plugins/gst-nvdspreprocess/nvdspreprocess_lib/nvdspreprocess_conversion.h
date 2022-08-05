/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * @file nvdspreprocess_conversion.h
 * <b>NVIDIA DeepStream Preprocess lib implementation </b>
 *
 * @b Description: This file contains cuda kernels used for custom
 * tensor preparation after normalization and mean subtraction for 2d conv
 * NCHW/NHWC models.
 */

/** @defgroup   gstreamer_nvdspreprocess_api NvDsPreProcess Plugin
 * Defines an API for the GStreamer NvDsPreProcess custom lib implementation.
 * @ingroup custom_gstreamer
 * @{
 */

#ifndef __NVDSPREPROCESS_CONVERSION_H__
#define __NVDSPREPROCESS_CONVERSION_H__

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * planar 3-channel float buffer of width x height resolution. The input buffer can
 * have a pitch > (width * 3). The cuda kernel supports normalization and mean
 * image subtraction.
 *
 * This kernel can be used for RGB -> RGB and BGR -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void NvDsPreProcessConvert_C3ToP3Float(float *outBuffer,
                                       unsigned char *inBuffer,
                                       unsigned int width,
                                       unsigned int height,
                                       unsigned int pitch,
                                       float scaleFactor,
                                       float *meanDataBuffer,
                                       cudaStream_t stream);

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * linear 3-channel float buffer of width x height resolution. The input buffer can
 * have a pitch > (width * 3). The cuda kernel supports normalization and mean
 * image subtraction.
 *
 * This kernel can be used for RGB -> RGB and BGR -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void NvDsPreProcessConvert_C3ToL3Float(float *outBuffer,
                                       unsigned char *inBuffer,
                                       unsigned int width,
                                       unsigned int height,
                                       unsigned int pitch,
                                       float scaleFactor,
                                       float *meanDataBuffer,
                                       cudaStream_t stream);

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * planar 3-channel float buffer of width x height resolution. The input buffer can
 * have a pitch > (width * 3). The cuda kernel supports normalization and mean
 * image subtraction.
 *
 * This kernel can be used for RGBA -> RGB and BGRx -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void NvDsPreProcessConvert_C4ToP3Float(float *outBuffer,
                                       unsigned char *inBuffer,
                                       unsigned int width,
                                       unsigned int height,
                                       unsigned int pitch,
                                       float scaleFactor,
                                       float *meanDataBuffer,
                                       cudaStream_t stream);

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * linear 3-channel float buffer of width x height resolution. The input buffer can
 * have a pitch > (width * 3). The cuda kernel supports normalization and mean
 * image subtraction.
 *
 * This kernel can be used for RGBA -> RGB and BGRx -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for linear float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void NvDsPreProcessConvert_C4ToL3Float(float *outBuffer,
                                       unsigned char *inBuffer,
                                       unsigned int width,
                                       unsigned int height,
                                       unsigned int pitch,
                                       float scaleFactor,
                                       float *meanDataBuffer,
                                       cudaStream_t stream);

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * planar 3-channel float buffer of width x height resolution with plane order
 * reversed. The input buffer can have a pitch > (width * 3). The cuda kernel
 * supports normalization and mean image subtraction.
 *
 * This kernel can be used for BGR -> RGB and RGB -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void NvDsPreProcessConvert_C3ToP3RFloat(float *outBuffer,
                                        unsigned char *inBuffer,
                                        unsigned int width,
                                        unsigned int height,
                                        unsigned int pitch,
                                        float scaleFactor,
                                        float *meanDataBuffer,
                                        cudaStream_t stream);

/**
 * Converts an input packed 3 channel buffer of width x height resolution into an
 * linear 3-channel float buffer of width x height resolution with plane order
 * reversed. The input buffer can have a pitch > (width * 3). The cuda kernel
 * supports normalization and mean image subtraction.
 *
 * This kernel can be used for BGR -> RGB and RGB -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void NvDsPreProcessConvert_C3ToL3RFloat(float *outBuffer,
                                        unsigned char *inBuffer,
                                        unsigned int width,
                                        unsigned int height,
                                        unsigned int pitch,
                                        float scaleFactor,
                                        float *meanDataBuffer,
                                        cudaStream_t stream);

/**
 * Converts an input packed 4 channel buffer of width x height resolution into an
 * planar 3-channel float buffer of width x height resolution with plane order
 * reversed. The input buffer can have a pitch > (width * 3). The cuda kernel
 * supports normalization and mean image subtraction.
 *
 * This kernel can be used for BGRx -> RGB and RGBA -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void NvDsPreProcessConvert_C4ToP3RFloat(float *outBuffer,
                                        unsigned char *inBuffer,
                                        unsigned int width,
                                        unsigned int height,
                                        unsigned int pitch,
                                        float scaleFactor,
                                        float *meanDataBuffer,
                                        cudaStream_t stream);

/**
 * Converts an input packed 4 channel buffer of width x height resolution into an
 * linear 3-channel float buffer of width x height resolution with plane order
 * reversed. The input buffer can have a pitch > (width * 3). The cuda kernel
 * supports normalization and mean image subtraction.
 *
 * This kernel can be used for BGRx -> RGB and RGBA -> BGR conversions.
 *
 * @param outBuffer      Cuda device buffer for planar float output. Should
 *                       be at least (width * height * 3 * sizeof(float)) bytes.
 * @param inBuffer       Cuda device buffer for packed input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * 3 * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void NvDsPreProcessConvert_C4ToL3RFloat(float *outBuffer,
                                        unsigned char *inBuffer,
                                        unsigned int width,
                                        unsigned int height,
                                        unsigned int pitch,
                                        float scaleFactor,
                                        float *meanDataBuffer,
                                        cudaStream_t stream);

/**
 * Converts an 1 channel UINT8 input of width x height resolution into an
 * 1 channel float buffer of width x height resolution. The input buffer can
 * have a pitch > width . The cuda kernel supports normalization and mean
 * image subtraction.
 *
 * @param outBuffer  Cuda device buffer for float output. Should
 *                       be at least (width * height * sizeof(float)) bytes.
 * @param inBuffer   Cuda device buffer for UINT8 input. Should be
 *                       at least (pitch * height) bytes.
 * @param width          Width of the buffers in pixels.
 * @param height         Height of the buffers in pixels.
 * @param pitch          Pitch of the input  buffer in bytes.
 * @param scaleFactor    Normalization factor.
 * @param meanDataBuffer Mean Image Data buffer. Should be at least
 *                       (width * height * sizeof(float)) bytes.
 * @param stream         Cuda stream identifier.
 */
void NvDsPreProcessConvert_C1ToP1Float(float *outBuffer,
                                       unsigned char *inBuffer,
                                       unsigned int width,
                                       unsigned int height,
                                       unsigned int pitch,
                                       float scaleFactor,
                                       float *meanDataBuffer,
                                       cudaStream_t stream);

void NvDsPreProcessConvert_FtFTensor(float *outBuffer,
                                     float *inBuffer,
                                     unsigned int width,
                                     unsigned int height,
                                     unsigned int pitch,
                                     float scaleFactor,
                                     float *meanDataBuffer,
                                     cudaStream_t stream);

/**
 * Function pointer type to which any of the NvDsPreProcessConvert functions can be
 * assigned.
 */
typedef void (*NvDsPreProcessConvertFcn)(float *outBuffer,
                                         unsigned char *inBuffer,
                                         unsigned int width,
                                         unsigned int height,
                                         unsigned int pitch,
                                         float scaleFactor,
                                         float *meanDataBuffer,
                                         cudaStream_t stream);

#endif /* __NVDSPREPROCESS_CONVERSION_H__ */
