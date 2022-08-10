/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <deepstream_config.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "capture_time_rules.h"
#include "gst-nvmessage.h"
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "nvds_obj_encode.h"

class ImageMetaConsumer {
public:
    enum ImageSizeType { FULL_FRAME, CROPPED_TO_OBJECT };

    /// Init an object and set that the queue is stopped
    ImageMetaConsumer();

    /// End the job of the thread dequeing by calling stop()
    ~ImageMetaConsumer();

    /// Fill information in the Consumer. This has to be done when these
    /// informations available.
    /// @param [in] output_folder_path Path where all the metadata will be saved.
    /// @param [in] frame_to_skip_rules_path Path where the rules for the amount
    /// of time to skip between 2 frames is stored.
    /// @param [in] min_box_confidence Minimum confidence that an object detected
    /// on a frame should have, in order to be kept.
    /// @param [in] max_box_confidence Maximum confidence that an object detected
    /// on a frame should have, in order to be kept.
    /// @param [in] min_box_width Minimum box width that an object detected
    /// on a frame should have, in order to be kept.
    /// @param [in] min_box_height Minimum box height that an object detected
    /// on a frame should have, in order to be kept.
    /// @param [in] save_full_frame_enabled Enable/Disable the save of complete
    /// images.
    /// @param [in] save_cropped_obj_enabled Enable/Disable the save of cropped
    /// images
    /// @param [in] seconds_to_skip_interval Unsigned integer giving the number of
    /// seconds to skip.
    /// @param [in] quality Quality.
    /// @param [in] source_nb Unsigned integer giving the number of sources that
    /// are currently giving a stream.
    void init(const std::string &output_folder_path,
              const std::string &frame_to_skip_rules_path,
              float min_box_confidence,
              float max_box_confidence,
              unsigned min_box_width,
              unsigned min_box_height,
              bool save_full_frame_enabled,
              bool save_cropped_obj_enabled,
              unsigned seconds_to_skip_interval,
              unsigned quality,
              unsigned source_nb);

    /// End the job of the current thread reading from the queue.
    void stop();

    /// Min confidence getter.
    /// @return The minimum confidence required for an object.
    float get_min_confidence() const;

    /// Max confidence getter.
    /// @return The maximum confidence required for an object.
    float get_max_confidence() const;

    /// Minimum box width getter.
    /// @return The minimum box width required for an object.
    unsigned get_min_box_width() const;

    /// Minimum box height getter.
    /// @return The minimum box height required for an object.
    unsigned get_min_box_height() const;

    /// Dequeuing thread status getter.
    /// @return The status of the thread reading metadata.
    bool get_is_stopped() const;

    /// Dequeuing full frame enabled getter.
    /// @return If complete images must be saved.
    bool get_save_full_frame_enabled() const;

    /// Dequeuing cropped image enable getter.
    /// @return If cropped images must be saved.
    bool get_save_cropped_images_enabled() const;

    /// Quality getter.
    /// @return The quality required save image.
    unsigned get_quality() const;

    /// Image Save Thread Handler
    /// @return the thread handler
    NvDsObjEncCtxHandle get_obj_ctx_handle();

    /// Metadata writer for a unique file (Json or Csv)
    /// @param extension of file requested (Json or Csv)
    /// @param stream_source_id Unique number identifying the stream source
    /// @param datetime_iso8601 current datetime formatted to iso 8601
    std::string make_img_path(ImageMetaConsumer::ImageSizeType ist,
                              unsigned stream_source_id,
                              const std::string &datetime_iso8601);

    /// Create image saving context if needed
    void init_image_save_library_on_first_time();

    /// This function checks if a certain amount of time passed before the
    /// last image was saved.
    /// \param source_id video stream number to check
    /// \return True if the required amount of time has passed before last
    /// the image was saved.
    bool should_save_data(unsigned source_id);

    /// If an image was saved, then the timestamp of the last image saved is
    /// stored. \param source_id Stream number for which the timestamp is saved.
    void data_was_saved_for_source(unsigned source_id);

private:
    /// Creates folder for images and metadata output.
    bool setup_folders();

    /// Creates a unique id for the current consumer.
    unsigned int get_unique_id();

    std::atomic<bool> is_stopped_;
    std::string output_folder_path_;
    std::string images_cropped_obj_output_folder_;
    std::string images_full_frame_output_folder_;
    std::mutex mutex_unique_index_;
    unsigned int unique_index_ = 0;
    float min_confidence_;
    float max_confidence_;
    unsigned min_box_width_;
    unsigned min_box_height_;
    bool save_full_frame_enabled_;
    bool save_cropped_obj_enabled_;
    unsigned quality_;
    std::vector<std::chrono::_V2::system_clock::time_point> time_last_frame_saved_list_;
    std::array<std::mutex, MAX_SOURCE_BINS> mutex_frame_saved_list_;
    CaptureTimeRules ctr_;
    NvDsObjEncCtxHandle obj_ctx_handle_;
    bool image_saving_library_is_init_;
};
