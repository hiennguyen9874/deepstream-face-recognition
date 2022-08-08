/**
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

#include "nvdscustomlib_base.h"

DSCustomLibraryBase::DSCustomLibraryBase()
{
    m_element = NULL;
    m_inCaps = NULL;
    m_outCaps = NULL;
    m_gpuId = 0;
}

bool DSCustomLibraryBase::SetInitParams(DSCustom_CreateParams *params)
{
    m_element = params->m_element;
    m_inCaps = params->m_inCaps;
    m_outCaps = params->m_outCaps;
    m_gpuId = params->m_gpuId;

    gst_audio_info_from_caps(&m_inAudioInfo, m_inCaps);
    gst_audio_info_from_caps(&m_outAudioInfo, m_outCaps);

    m_inAudioFmt = GST_AUDIO_FORMAT_INFO_FORMAT(m_inAudioInfo.finfo);
    m_outAudioFmt = GST_AUDIO_FORMAT_INFO_FORMAT(m_outAudioInfo.finfo);

    return true;
}

DSCustomLibraryBase::~DSCustomLibraryBase()
{
}

GstCaps *DSCustomLibraryBase::GetCompatibleCaps(GstPadDirection direction,
                                                GstCaps *in_caps,
                                                GstCaps *othercaps)
{
    GstCaps *result = gst_caps_copy(in_caps);
    return result;
}