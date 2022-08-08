/**
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
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

#include "dsexample_lib.h"

#include <stdio.h>
#include <stdlib.h>

struct DsExampleCtx {
    DsExampleInitParams initParams;
};

DsExampleCtx *DsExampleCtxInit(DsExampleInitParams *initParams)
{
    DsExampleCtx *ctx = (DsExampleCtx *)calloc(1, sizeof(DsExampleCtx));
    ctx->initParams = *initParams;
    return ctx;
}

// In case of an actual processing library, processing on data wil be completed
// in this function and output will be returned
DsExampleOutput *DsExampleProcess(DsExampleCtx *ctx, unsigned char *data)
{
    DsExampleOutput *out = (DsExampleOutput *)calloc(1, sizeof(DsExampleOutput));

    if (data != NULL) {
        // Process your data here
    }
    // Fill output structure using processed output
    // Here, we fake some detected objects and labels
    if (ctx->initParams.fullFrame) {
        out->numObjects = 2;
        out->object[0] = (DsExampleObject){(float)(ctx->initParams.processingWidth) / 8,
                                           (float)(ctx->initParams.processingHeight) / 8,
                                           (float)(ctx->initParams.processingWidth) / 8,
                                           (float)(ctx->initParams.processingHeight) / 8, "Obj0"};

        out->object[1] = (DsExampleObject){(float)(ctx->initParams.processingWidth) / 2,
                                           (float)(ctx->initParams.processingHeight) / 2,
                                           (float)(ctx->initParams.processingWidth) / 8,
                                           (float)(ctx->initParams.processingHeight) / 8, "Obj1"};
    } else {
        out->numObjects = 1;
        out->object[0] = (DsExampleObject){(float)(ctx->initParams.processingWidth) / 8,
                                           (float)(ctx->initParams.processingHeight) / 8,
                                           (float)(ctx->initParams.processingWidth) / 8,
                                           (float)(ctx->initParams.processingHeight) / 8, ""};
        // Set the object label
        snprintf(out->object[0].label, 64, "Obj_label");
    }

    return out;
}

void DsExampleCtxDeinit(DsExampleCtx *ctx)
{
    free(ctx);
}
