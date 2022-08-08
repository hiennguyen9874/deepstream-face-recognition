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

#ifndef __NVDS_TTS_CUSTOMLIB_BASE_HPP__
#define __NVDS_TTS_CUSTOMLIB_BASE_HPP__

#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>

#include "nvdscustomlib_interface.hpp"

namespace nvdstts {

class DSCustomLibraryBase : public IDSCustomLibrary {
public:
    DSCustomLibraryBase() = default;
    virtual ~DSCustomLibraryBase() override;

    bool Initialize() override;

    /* Set Init Parameters */
    bool StartWithParams(DSCustom_CreateParams *params) override;

    /* Set Each Property */
    bool SetProperty(const Property &prop) override;

    /* Get Compatible Input/Output Caps */
    GstCaps *GetCompatibleCaps(GstPadDirection direction,
                               GstCaps *inCaps,
                               GstCaps *otherCaps) override;

    /* Handle event, e.g. EOS... */
    bool HandleEvent(GstEvent *event) override { return true; }

    /* Process Input Buffer */
    BufferResult ProcessBuffer(GstBuffer *inbuf) override = 0;

protected:
    /* Gstreamer dstts plugin's base class reference */
    GstBaseTransform *m_element{nullptr};
    /* Gst Caps Information */
    GstCaps *m_inCaps{nullptr};
    GstCaps *m_outCaps{nullptr};
    std::string m_configFile;

    /* Audio Information */
    /* Output Information */
    CapsType m_OutType = CapsType::kNone;
    GstAudioInfo m_outAudioInfo{nullptr, GST_AUDIO_FLAG_NONE};
    GstAudioFormat m_outAudioFmt = GST_AUDIO_FORMAT_UNKNOWN;
};

} // namespace nvdstts

#endif
