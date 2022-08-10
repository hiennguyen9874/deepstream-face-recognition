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

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

class CaptureTimeRules {
    typedef std::chrono::time_point<std::chrono::system_clock> t_time_pt;
    typedef std::chrono::duration<unsigned long long> t_duration;

public:
    /// Fills the time rules with the content of the file in path
    /// \param path containing the time rules
    /// \param default_second_interval When no rules are present for a
    /// certain time interval, this default duration is used.
    void init(const std::string &path, unsigned default_second_interval);

    /// Compute the correct time interval using the local computer time.
    /// \return the computed time interval to skip for current time
    t_duration getCurrentTimeInterval();

    /// \return True if the construction of the the object went well.
    /// False otherwise
    bool is_init_();

private:
    struct TimeRule {
        unsigned begin_time_hour;
        unsigned begin_time_minute;
        unsigned end_time_hour;
        unsigned end_time_minute;
        unsigned interval_between_frame_capture_seconds;
        bool end_time_is_next_day;
    };

    /// Test if a timestamp is between 2 other timestamps
    /// \param t structure containing the left and right bounding
    /// bounding timestamps
    /// \param now the current time
    /// \return True if the current time is between the 2 bounding
    /// timestamps
    static bool isInTimeRule(const TimeRule &t, const tm &now);

    enum ParseResult {
        PARSE_RESULT_OK,
        PARSE_RESULT_BAD_CHARS,
        PARSE_RESULT_OUT_OF_BOUND,
        PARSE_RESULT_EMPTY
    };

    static ParseResult stoi_err_handling(unsigned &dst, const std::string &src, unsigned max_bound);
    static bool parsing_contains_error(const std::vector<ParseResult> &parse_res_list,
                                       const std::vector<std::string> &str_list,
                                       const std::string &curr_line,
                                       unsigned line_number);
    bool single_time_rule_parser(const std::string &path,
                                 const std::string &line,
                                 unsigned line_number);

    std::chrono::seconds default_duration_;
    t_time_pt end_of_current_time_interval_;
    t_duration current_time_interval_;
    std::vector<TimeRule> rules_;
    bool init_ = false;
};
