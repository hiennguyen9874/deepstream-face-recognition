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

#include <cuda.h>

#include "nvdspreprocess_conversion.h"

#define THREADS_PER_BLOCK 32
#define THREADS_PER_BLOCK_1 (THREADS_PER_BLOCK - 1)

__global__ void NvDsPreProcessConvert_CxToP3FloatKernel(float *outBuffer,
                                                        unsigned char *inBuffer,
                                                        unsigned int width,
                                                        unsigned int height,
                                                        unsigned int pitch,
                                                        unsigned int inputPixelSize,
                                                        float scaleFactor)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        for (unsigned int k = 0; k < 3; k++) {
            outBuffer[width * height * k + row * width + col] =
                scaleFactor * inBuffer[row * pitch + col * inputPixelSize + k];
        }
    }
}

__global__ void NvDsPreProcessConvert_CxToL3FloatKernel(float *outBuffer,
                                                        unsigned char *inBuffer,
                                                        unsigned int width,
                                                        unsigned int height,
                                                        unsigned int pitch,
                                                        unsigned int inputPixelSize,
                                                        float scaleFactor)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        for (unsigned int k = 0; k < 3; k++) {
            outBuffer[row * width * 3 + col * 3 + k] =
                scaleFactor * inBuffer[row * pitch + col * inputPixelSize + k];
        }
    }
}

__global__ void NvDsPreProcessConvert_CxToP3FloatKernelWithMeanSubtraction(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor,
    float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        for (unsigned int k = 0; k < 3; k++) {
            outBuffer[width * height * k + row * width + col] =
                scaleFactor * ((float)inBuffer[row * pitch + col * inputPixelSize + k] -
                               meanDataBuffer[(row * width * 3) + (col * 3) + k]);
        }
    }
}

__global__ void NvDsPreProcessConvert_CxToL3FloatKernelWithMeanSubtraction(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor,
    float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        for (unsigned int k = 0; k < 3; k++) {
            outBuffer[row * width * 3 + col * 3 + k] =
                scaleFactor * ((float)inBuffer[row * pitch + col * inputPixelSize + k] -
                               meanDataBuffer[(row * width * 3) + (col * 3) + k]);
        }
    }
}

__global__ void NvDsPreProcessConvert_CxToP3RFloatKernel(float *outBuffer,
                                                         unsigned char *inBuffer,
                                                         unsigned int width,
                                                         unsigned int height,
                                                         unsigned int pitch,
                                                         unsigned int inputPixelSize,
                                                         float scaleFactor)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        for (unsigned int k = 0; k < 3; k++) {
            outBuffer[width * height * k + row * width + col] =
                scaleFactor * inBuffer[row * pitch + col * inputPixelSize + (2 - k)];
        }
    }
}

__global__ void NvDsPreProcessConvert_CxToL3RFloatKernel(float *outBuffer,
                                                         unsigned char *inBuffer,
                                                         unsigned int width,
                                                         unsigned int height,
                                                         unsigned int pitch,
                                                         unsigned int inputPixelSize,
                                                         float scaleFactor)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        for (unsigned int k = 0; k < 3; k++) {
            outBuffer[row * width * 3 + col * 3 + k] =
                scaleFactor * inBuffer[row * pitch + col * inputPixelSize + (2 - k)];
        }
    }
}

__global__ void NvDsPreProcessConvert_CxToP3RFloatKernelWithMeanSubtraction(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor,
    float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        for (unsigned int k = 0; k < 3; k++) {
            outBuffer[width * height * k + row * width + col] =
                scaleFactor * ((float)inBuffer[row * pitch + col * inputPixelSize + (2 - k)] -
                               meanDataBuffer[(row * width * 3) + (col * 3) + k]);
        }
    }
}

__global__ void NvDsPreProcessConvert_CxToL3RFloatKernelWithMeanSubtraction(
    float *outBuffer,
    unsigned char *inBuffer,
    unsigned int width,
    unsigned int height,
    unsigned int pitch,
    unsigned int inputPixelSize,
    float scaleFactor,
    float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        for (unsigned int k = 0; k < 3; k++) {
            outBuffer[row * width * 3 + col * 3 + k] =
                scaleFactor * ((float)inBuffer[row * pitch + col * inputPixelSize + (2 - k)] -
                               meanDataBuffer[(row * width * 3) + (col * 3) + k]);
        }
    }
}

__global__ void NvDsPreProcessConvert_C1ToP1FloatKernel(float *outBuffer,
                                                        unsigned char *inBuffer,
                                                        unsigned int width,
                                                        unsigned int height,
                                                        unsigned int pitch,
                                                        float scaleFactor)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        outBuffer[row * width + col] = scaleFactor * inBuffer[row * pitch + col];
    }
}

__global__ void NvDsPreProcessConvert_C1ToP1FloatKernelWithMeanSubtraction(float *outBuffer,
                                                                           unsigned char *inBuffer,
                                                                           unsigned int width,
                                                                           unsigned int height,
                                                                           unsigned int pitch,
                                                                           float scaleFactor,
                                                                           float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        outBuffer[row * width + col] = scaleFactor * ((float)inBuffer[row * pitch + col] -
                                                      meanDataBuffer[(row * width) + col]);
    }
}

__global__ void NvDsPreProcessConvert_FtFTensorKernel(float *outBuffer,
                                                      float *inBuffer,
                                                      unsigned int width,
                                                      unsigned int height,
                                                      unsigned int pitch,
                                                      float scaleFactor)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        outBuffer[row * width + col] = scaleFactor * inBuffer[row * width + col];
    }
}

__global__ void NvDsPreProcessConvert_FtFTensorKernelWithMeanSubtraction(float *outBuffer,
                                                                         float *inBuffer,
                                                                         unsigned int width,
                                                                         unsigned int height,
                                                                         unsigned int pitch,
                                                                         float scaleFactor,
                                                                         float *meanDataBuffer)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        outBuffer[row * width + col] = scaleFactor * ((float)inBuffer[row * width + col] -
                                                      meanDataBuffer[(row * width) + col]);
    }
}

void NvDsPreProcessConvert_C3ToP3Float(float *outBuffer,
                                       unsigned char *inBuffer,
                                       unsigned int width,
                                       unsigned int height,
                                       unsigned int pitch,
                                       float scaleFactor,
                                       float *meanDataBuffer,
                                       cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width + THREADS_PER_BLOCK_1) / threadsPerBlock.x,
                (height + THREADS_PER_BLOCK_1) / threadsPerBlock.y);

    if (meanDataBuffer == NULL) {
        NvDsPreProcessConvert_CxToP3FloatKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            outBuffer, inBuffer, width, height, pitch, 3, scaleFactor);
    } else {
        NvDsPreProcessConvert_CxToP3FloatKernelWithMeanSubtraction<<<blocks, threadsPerBlock, 0,
                                                                     stream>>>(
            outBuffer, inBuffer, width, height, pitch, 3, scaleFactor, meanDataBuffer);
    }
}

void NvDsPreProcessConvert_C3ToL3Float(float *outBuffer,
                                       unsigned char *inBuffer,
                                       unsigned int width,
                                       unsigned int height,
                                       unsigned int pitch,
                                       float scaleFactor,
                                       float *meanDataBuffer,
                                       cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width + THREADS_PER_BLOCK_1) / threadsPerBlock.x,
                (height + THREADS_PER_BLOCK_1) / threadsPerBlock.y);

    if (meanDataBuffer == NULL) {
        NvDsPreProcessConvert_CxToL3FloatKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            outBuffer, inBuffer, width, height, pitch, 3, scaleFactor);
    } else {
        NvDsPreProcessConvert_CxToL3FloatKernelWithMeanSubtraction<<<blocks, threadsPerBlock, 0,
                                                                     stream>>>(
            outBuffer, inBuffer, width, height, pitch, 3, scaleFactor, meanDataBuffer);
    }
}

void NvDsPreProcessConvert_C4ToP3Float(float *outBuffer,
                                       unsigned char *inBuffer,
                                       unsigned int width,
                                       unsigned int height,
                                       unsigned int pitch,
                                       float scaleFactor,
                                       float *meanDataBuffer,
                                       cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width + THREADS_PER_BLOCK_1) / threadsPerBlock.x,
                (height + THREADS_PER_BLOCK_1) / threadsPerBlock.y);

    if (meanDataBuffer == NULL) {
        NvDsPreProcessConvert_CxToP3FloatKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            outBuffer, inBuffer, width, height, pitch, 4, scaleFactor);
    } else {
        NvDsPreProcessConvert_CxToP3FloatKernelWithMeanSubtraction<<<blocks, threadsPerBlock, 0,
                                                                     stream>>>(
            outBuffer, inBuffer, width, height, pitch, 4, scaleFactor, meanDataBuffer);
    }
}

void NvDsPreProcessConvert_C4ToL3Float(float *outBuffer,
                                       unsigned char *inBuffer,
                                       unsigned int width,
                                       unsigned int height,
                                       unsigned int pitch,
                                       float scaleFactor,
                                       float *meanDataBuffer,
                                       cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width + THREADS_PER_BLOCK_1) / threadsPerBlock.x,
                (height + THREADS_PER_BLOCK_1) / threadsPerBlock.y);

    if (meanDataBuffer == NULL) {
        NvDsPreProcessConvert_CxToL3FloatKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            outBuffer, inBuffer, width, height, pitch, 4, scaleFactor);
    } else {
        NvDsPreProcessConvert_CxToL3FloatKernelWithMeanSubtraction<<<blocks, threadsPerBlock, 0,
                                                                     stream>>>(
            outBuffer, inBuffer, width, height, pitch, 4, scaleFactor, meanDataBuffer);
    }
}

void NvDsPreProcessConvert_C3ToP3RFloat(float *outBuffer,
                                        unsigned char *inBuffer,
                                        unsigned int width,
                                        unsigned int height,
                                        unsigned int pitch,
                                        float scaleFactor,
                                        float *meanDataBuffer,
                                        cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width + THREADS_PER_BLOCK_1) / threadsPerBlock.x,
                (height + THREADS_PER_BLOCK_1) / threadsPerBlock.y);

    if (meanDataBuffer == NULL) {
        NvDsPreProcessConvert_CxToP3RFloatKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            outBuffer, inBuffer, width, height, pitch, 3, scaleFactor);
    } else {
        NvDsPreProcessConvert_CxToP3RFloatKernelWithMeanSubtraction<<<blocks, threadsPerBlock, 0,
                                                                      stream>>>(
            outBuffer, inBuffer, width, height, pitch, 3, scaleFactor, meanDataBuffer);
    }
}

void NvDsPreProcessConvert_C3ToL3RFloat(float *outBuffer,
                                        unsigned char *inBuffer,
                                        unsigned int width,
                                        unsigned int height,
                                        unsigned int pitch,
                                        float scaleFactor,
                                        float *meanDataBuffer,
                                        cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width + THREADS_PER_BLOCK_1) / threadsPerBlock.x,
                (height + THREADS_PER_BLOCK_1) / threadsPerBlock.y);

    if (meanDataBuffer == NULL) {
        NvDsPreProcessConvert_CxToL3RFloatKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            outBuffer, inBuffer, width, height, pitch, 3, scaleFactor);
    } else {
        NvDsPreProcessConvert_CxToL3RFloatKernelWithMeanSubtraction<<<blocks, threadsPerBlock, 0,
                                                                      stream>>>(
            outBuffer, inBuffer, width, height, pitch, 3, scaleFactor, meanDataBuffer);
    }
}

void NvDsPreProcessConvert_C4ToP3RFloat(float *outBuffer,
                                        unsigned char *inBuffer,
                                        unsigned int width,
                                        unsigned int height,
                                        unsigned int pitch,
                                        float scaleFactor,
                                        float *meanDataBuffer,
                                        cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width + THREADS_PER_BLOCK_1) / threadsPerBlock.x,
                (height + THREADS_PER_BLOCK_1) / threadsPerBlock.y);

    if (meanDataBuffer == NULL) {
        NvDsPreProcessConvert_CxToP3RFloatKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            outBuffer, inBuffer, width, height, pitch, 4, scaleFactor);
    } else {
        NvDsPreProcessConvert_CxToP3RFloatKernelWithMeanSubtraction<<<blocks, threadsPerBlock, 0,
                                                                      stream>>>(
            outBuffer, inBuffer, width, height, pitch, 4, scaleFactor, meanDataBuffer);
    }
}

void NvDsPreProcessConvert_C4ToL3RFloat(float *outBuffer,
                                        unsigned char *inBuffer,
                                        unsigned int width,
                                        unsigned int height,
                                        unsigned int pitch,
                                        float scaleFactor,
                                        float *meanDataBuffer,
                                        cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width + THREADS_PER_BLOCK_1) / threadsPerBlock.x,
                (height + THREADS_PER_BLOCK_1) / threadsPerBlock.y);

    if (meanDataBuffer == NULL) {
        NvDsPreProcessConvert_CxToL3RFloatKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            outBuffer, inBuffer, width, height, pitch, 4, scaleFactor);
    } else {
        NvDsPreProcessConvert_CxToL3RFloatKernelWithMeanSubtraction<<<blocks, threadsPerBlock, 0,
                                                                      stream>>>(
            outBuffer, inBuffer, width, height, pitch, 4, scaleFactor, meanDataBuffer);
    }
}

void NvDsPreProcessConvert_C1ToP1Float(float *outBuffer,
                                       unsigned char *inBuffer,
                                       unsigned int width,
                                       unsigned int height,
                                       unsigned int pitch,
                                       float scaleFactor,
                                       float *meanDataBuffer,
                                       cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width + THREADS_PER_BLOCK_1) / threadsPerBlock.x,
                (height + THREADS_PER_BLOCK_1) / threadsPerBlock.y);

    if (meanDataBuffer == NULL) {
        NvDsPreProcessConvert_C1ToP1FloatKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            outBuffer, inBuffer, width, height, pitch, scaleFactor);
    } else {
        NvDsPreProcessConvert_C1ToP1FloatKernelWithMeanSubtraction<<<blocks, threadsPerBlock, 0,
                                                                     stream>>>(
            outBuffer, inBuffer, width, height, pitch, scaleFactor, meanDataBuffer);
    }
}

// TODO add channel information, current implementation is only for single channel
void NvDsPreProcessConvert_FtFTensor(float *outBuffer,
                                     float *inBuffer,
                                     unsigned int width,
                                     unsigned int height,
                                     unsigned int pitch,
                                     float scaleFactor,
                                     float *meanDataBuffer,
                                     cudaStream_t stream)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width + THREADS_PER_BLOCK_1) / threadsPerBlock.x,
                (height + THREADS_PER_BLOCK_1) / threadsPerBlock.y);

    if (meanDataBuffer == NULL) {
        NvDsPreProcessConvert_FtFTensorKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            outBuffer, inBuffer, width, height, pitch, scaleFactor);
    } else {
        NvDsPreProcessConvert_FtFTensorKernelWithMeanSubtraction<<<blocks, threadsPerBlock, 0,
                                                                   stream>>>(
            outBuffer, inBuffer, width, height, pitch, scaleFactor, meanDataBuffer);
    }
}
