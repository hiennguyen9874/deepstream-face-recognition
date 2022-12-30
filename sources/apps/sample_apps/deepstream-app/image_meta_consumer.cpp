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

#include "image_meta_consumer.h"

constexpr unsigned seconds_in_one_day = 86400;

static int is_dir(const char *path)
{
    struct stat path_stat;
    stat(path, &path_stat);
    return S_ISDIR(path_stat.st_mode);
}

ImageMetaConsumer::ImageMetaConsumer()
    : is_stopped_(true), save_full_frame_enabled_(true), save_cropped_obj_enabled_(false),
      obj_ctx_handle_((NvDsObjEncCtxHandle)0), image_saving_library_is_init_(false)
{
}

ImageMetaConsumer::~ImageMetaConsumer()
{
    stop();
}

unsigned int ImageMetaConsumer::get_unique_id()
{
    mutex_unique_index_.lock();
    auto res = unique_index_++;
    mutex_unique_index_.unlock();
    return res;
}

void ImageMetaConsumer::stop()
{
    if (is_stopped_)
        return;
    is_stopped_ = true;
    if (image_saving_library_is_init_)
        nvds_obj_enc_destroy_context(obj_ctx_handle_);
}

void ImageMetaConsumer::init(const std::string &output_folder_path,
                             const std::string &frame_to_skip_rules_path,
                             const float min_box_confidence,
                             const float max_box_confidence,
                             const unsigned min_box_width,
                             const unsigned min_box_height,
                             const bool save_full_frame_enabled,
                             const bool save_cropped_obj_enabled,
                             const unsigned seconds_to_skip_interval,
                             const unsigned quality,
                             const unsigned source_nb)
{
    if (!is_stopped_) {
        std::cerr << "Consummer already running.\n";
        return;
    }
    if (!is_dir(output_folder_path.c_str())) {
        std::cerr << "Missing directory: " << output_folder_path << ".\n";
        return;
    }
    output_folder_path_ = output_folder_path;
    if (output_folder_path_.back() != '/')
        output_folder_path_ += '/';

    ctr_.init(frame_to_skip_rules_path, seconds_to_skip_interval);
    if (!ctr_.is_init_())
        return;

    if (!setup_folders()) {
        std::cerr << "Could not create " << images_cropped_obj_output_folder_ << " and "
                  << images_full_frame_output_folder_ << "\n";
        return;
    }

    min_confidence_ = min_box_confidence;
    max_confidence_ = max_box_confidence;
    min_box_width_ = min_box_width;
    min_box_height_ = min_box_height;
    save_full_frame_enabled_ = save_full_frame_enabled;
    save_cropped_obj_enabled_ = save_cropped_obj_enabled;
    quality_ = quality;

    auto stsi = std::chrono::seconds(seconds_in_one_day);
    for (unsigned i = 0; i < source_nb; ++i)
        time_last_frame_saved_list_.push_back(std::chrono::system_clock::now() - stsi);

    is_stopped_ = false;
}

bool ImageMetaConsumer::setup_folders()
{
    images_cropped_obj_output_folder_ = output_folder_path_ + "images_cropped/";
    images_full_frame_output_folder_ = output_folder_path_ + "images/";
    unsigned permissions = 0755;
    mkdir(images_cropped_obj_output_folder_.c_str(), permissions);
    mkdir(images_full_frame_output_folder_.c_str(), permissions);
    return is_dir(images_cropped_obj_output_folder_.c_str()) &&
           is_dir(images_full_frame_output_folder_.c_str());
}

std::string ImageMetaConsumer::make_img_path(const ImageMetaConsumer::ImageSizeType ist,
                                             const unsigned stream_source_id,
                                             const std::string &datetime_iso8601)
{
    unsigned long id = get_unique_id();
    std::stringstream ss;
    switch (ist) {
    case FULL_FRAME:
        ss << images_full_frame_output_folder_;
        break;
    case CROPPED_TO_OBJECT:
        ss << images_cropped_obj_output_folder_;
        break;
    }
    ss << "camera-" << stream_source_id << "_";
    ss << datetime_iso8601 << "_";
    ss << std::setw(10) << std::setfill('0') << id;
    ss << ".jpg";
    return ss.str();
}

NvDsObjEncCtxHandle ImageMetaConsumer::get_obj_ctx_handle()
{
    return obj_ctx_handle_;
}

float ImageMetaConsumer::get_min_confidence() const
{
    return min_confidence_;
}

float ImageMetaConsumer::get_max_confidence() const
{
    return max_confidence_;
}

unsigned ImageMetaConsumer::get_min_box_width() const
{
    return min_box_width_;
}

unsigned ImageMetaConsumer::get_min_box_height() const
{
    return min_box_height_;
}

bool ImageMetaConsumer::get_is_stopped() const
{
    return is_stopped_;
}

bool ImageMetaConsumer::get_save_full_frame_enabled() const
{
    return save_full_frame_enabled_;
}

bool ImageMetaConsumer::get_save_cropped_images_enabled() const
{
    return save_cropped_obj_enabled_;
}

unsigned ImageMetaConsumer::get_quality() const
{
    return quality_;
}

void ImageMetaConsumer::init_image_save_library_on_first_time()
{
    if (!image_saving_library_is_init_) {
        if (!image_saving_library_is_init_ &&
            (save_cropped_obj_enabled_ || save_full_frame_enabled_)) {
            obj_ctx_handle_ = nvds_obj_enc_create_context();
            if (obj_ctx_handle_)
                image_saving_library_is_init_ = true;
            else
                std::cerr << "Unable to create encoding context\n";
        }
    }
}

bool ImageMetaConsumer::should_save_data(unsigned source_id)
{
    auto time_interval = ctr_.getCurrentTimeInterval();
    auto now = std::chrono::system_clock::now();
    if (now < time_last_frame_saved_list_[source_id]) {
        std::cerr << "The current time seems to have moved to the past\n"
                  << "Reseting time for last frame save for source" << source_id << "\n";
        time_last_frame_saved_list_[source_id] = now;
    }

    return now > time_interval + time_last_frame_saved_list_[source_id];
}

void ImageMetaConsumer::data_was_saved_for_source(unsigned source_id)
{
    time_last_frame_saved_list_[source_id] = std::chrono::system_clock::now();
}
