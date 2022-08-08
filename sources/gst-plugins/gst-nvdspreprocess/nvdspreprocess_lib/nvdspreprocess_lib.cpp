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

#include "nvdspreprocess_lib.h"

#include <string.h>

#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvdspreprocess_impl.h"

struct CustomCtx {
    /** Custom initialization parameters */
    CustomInitParams initParams;
    /** Custom mean subtraction and normalization parameters */
    CustomMeanSubandNormParams custom_mean_norm_params;
    /** unique pointer to tensor_impl class instance */
    std::unique_ptr<NvDsPreProcessTensorImpl> tensor_impl;
};

/* Get the absolute path of a file mentioned in the config given a
 * file path absolute/relative to the config file. */
static gboolean get_absolute_file_path(const gchar *cfg_file_path,
                                       const gchar *file_path,
                                       char *abs_path_str)
{
    gchar abs_cfg_path[_PATH_MAX + 1];
    gchar abs_real_file_path[_PATH_MAX + 1];
    gchar *abs_file_path;
    gchar *delim;

    /* Absolute path. No need to resolve further. */
    if (file_path[0] == '/') {
        /* Check if the file exists, return error if not. */
        if (!realpath(file_path, abs_real_file_path)) {
            return FALSE;
        }
        g_strlcpy(abs_path_str, abs_real_file_path, _PATH_MAX);
        return TRUE;
    }

    /* Get the absolute path of the config file. */
    if (!realpath(cfg_file_path, abs_cfg_path)) {
        return FALSE;
    }

    /* Remove the file name from the absolute path to get the directory of the
     * config file. */
    delim = g_strrstr(abs_cfg_path, "/");
    *(delim + 1) = '\0';

    /* Get the absolute file path from the config file's directory path and
     * relative file path. */
    abs_file_path = g_strconcat(abs_cfg_path, file_path, nullptr);

    /* Resolve the path.*/
    if (realpath(abs_file_path, abs_real_file_path) == nullptr) {
        /* Ignore error if file does not exist and use the unresolved path. */
        if (errno == ENOENT)
            g_strlcpy(abs_real_file_path, abs_file_path, _PATH_MAX);
        else
            return FALSE;
    }

    g_free(abs_file_path);

    g_strlcpy(abs_path_str, abs_real_file_path, _PATH_MAX);
    return TRUE;
}

NvDsPreProcessStatus CustomTensorPreparation(CustomCtx *ctx,
                                             NvDsPreProcessBatch *batch,
                                             NvDsPreProcessCustomBuf *&buf,
                                             CustomTensorParams &tensorParam,
                                             NvDsPreProcessAcquirer *acquirer)
{
    NvDsPreProcessStatus status = NVDSPREPROCESS_TENSOR_NOT_READY;

    /** acquire a buffer from tensor pool */
    buf = acquirer->acquire();

    /** Prepare Tensor */
    status = ctx->tensor_impl->prepare_tensor(batch, buf->memory_ptr);
    if (status != NVDSPREPROCESS_SUCCESS) {
        printf("Custom Lib: Tensor Preparation failed\n");
        acquirer->release(buf);
    }

    /** synchronize cuda stream */
    status = ctx->tensor_impl->syncStream();
    if (status != NVDSPREPROCESS_SUCCESS) {
        printf("Custom Lib: Cuda Stream Synchronization failed\n");
        acquirer->release(buf);
    }

    tensorParam.params.network_input_shape[0] = (int)batch->units.size();

    if (status != NVDSPREPROCESS_SUCCESS) {
        printf("CustomTensorPreparation failed\n");
        acquirer->release(buf);
    }

    return status;
}

NvDsPreProcessStatus CustomTransformation(NvBufSurface *in_surf,
                                          NvBufSurface *out_surf,
                                          CustomTransformParams &params)
{
    NvBufSurfTransform_Error err;

    err = NvBufSurfTransformSetSessionParams(&params.transform_config_params);
    if (err != NvBufSurfTransformError_Success) {
        printf("NvBufSurfTransformSetSessionParams failed with error %d\n", err);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    /* Batched tranformation. */
    err = NvBufSurfTransform(in_surf, out_surf, &params.transform_params);

    if (err != NvBufSurfTransformError_Success) {
        printf("NvBufSurfTransform failed with error %d\n", err);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus CustomAsyncTransformation(NvBufSurface *in_surf,
                                               NvBufSurface *out_surf,
                                               CustomTransformParams &params)
{
    NvBufSurfTransform_Error err;

    err = NvBufSurfTransformSetSessionParams(&params.transform_config_params);
    if (err != NvBufSurfTransformError_Success) {
        printf("NvBufSurfTransformSetSessionParams failed with error %d\n", err);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    /* Async Batched tranformation. */
    err = NvBufSurfTransformAsync(in_surf, out_surf, &params.transform_params, &params.sync_obj);

    if (err != NvBufSurfTransformError_Success) {
        printf("NvBufSurfTransform failed with error %d\n", err);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }

    return NVDSPREPROCESS_SUCCESS;
}

CustomCtx *initLib(CustomInitParams initparams)
{
    CustomCtx *ctx = new CustomCtx;
    NvDsPreProcessStatus status;

    ctx->custom_mean_norm_params.pixel_normalization_factor =
        std::stof(initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_PIXEL_NORMALIZATION_FACTOR]);

    if (!initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_MEAN_FILE].empty()) {
        gchar *abs_path = new gchar[_PATH_MAX];
        if (!get_absolute_file_path(
                initparams.config_file_path,
                initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_MEAN_FILE].c_str(), abs_path)) {
            printf("Error: Could not parse mean image file path\n");
            return nullptr;
        }
        if (!ctx->custom_mean_norm_params.meanImageFilePath.empty()) {
            ctx->custom_mean_norm_params.meanImageFilePath.clear();
        }
        ctx->custom_mean_norm_params.meanImageFilePath.append(abs_path);
        delete abs_path;
    }

    std::string offsets_str = initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_OFFSETS];

    if (!offsets_str.empty()) {
        std::string delimiter = ";";
        size_t pos = 0;
        std::string token;

        while ((pos = offsets_str.find(delimiter)) != std::string::npos) {
            token = offsets_str.substr(0, pos);
            ctx->custom_mean_norm_params.offsets.push_back(std::stof(token));
            offsets_str.erase(0, pos + delimiter.length());
        }
        ctx->custom_mean_norm_params.offsets.push_back(std::stof(offsets_str));

        printf("Using offsets : %f,%f,%f\n", ctx->custom_mean_norm_params.offsets[0],
               ctx->custom_mean_norm_params.offsets[1], ctx->custom_mean_norm_params.offsets[2]);
    }

    status = normalization_mean_subtraction_impl_initialize(&ctx->custom_mean_norm_params,
                                                            &initparams.tensor_params,
                                                            ctx->tensor_impl, initparams.unique_id);

    if (status != NVDSPREPROCESS_SUCCESS) {
        printf("normalization_mean_subtraction_impl_initialize failed\n");
        return nullptr;
    }

    ctx->initParams = initparams;

    return ctx;
}

void deInitLib(CustomCtx *ctx)
{
    delete ctx;
}