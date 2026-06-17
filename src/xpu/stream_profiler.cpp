/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "xpu/stream_profiler.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
status_t verbose_profiler_t::add_to_pending_primitive_list(
        double start_ms, const std::string &pd_info) {
    if (events_.empty()) return status::invalid_arguments;

    // Adds metadata to the last entry
    auto &last_entry = events_.back();
    last_entry.start_ms = start_ms;
    last_entry.pd_info = pd_info;
    return status::success;
}

void verbose_profiler_t::check_for_completed_primitives() {

    for (auto it = events_.begin(); it != events_.end();) {
        const auto &prof_data = *it;
        auto &evts = prof_data.prim_events;
        double duration_ms = 0.0;

        // handles primitives with no kernels - here
        // device time is effectively zero
        if (evts.empty() && !prof_data.pd_info.empty()) {
            VPROF(prof_data.start_ms, primitive, exec, VERBOSE_profile,
                    prof_data.pd_info.c_str(), duration_ms);
            it = events_.erase(it);
            continue;
        }

        // avoids logging info for empty descriptors
        if (prof_data.pd_info.empty()) {
            it = events_.erase(it);
            continue;
        }

        // the polling check is paused at the first pending primitive
        // and resumed again at the next check. This ensures the
        // primitives are logged in the same order they were enqueued
        if (!is_event_complete(evts.back())) { break; }

        size_t index = std::distance(events_.begin(), it);
        status_t status = get_aggregate_exec_time(index, duration_ms);

        if (status == status::success) {
            VPROF(prof_data.start_ms, primitive, exec, VERBOSE_profile,
                    prof_data.pd_info.c_str(), duration_ms);
        } else {
            VERROR(common, runtime,
                    "%s, profiling error: failures in exec time computation",
                    prof_data.pd_info.c_str());
        }

        // the primitive info entry is removed after logging
        // to avoid blowing up the sizes of events_
        it = events_.erase(it);
    }
}

void verbose_profiler_t::wait_for_pending_primitives() {

    for (auto &prof_data : events_) {
        auto &evts = prof_data.prim_events;

        // For in-order queues, waiting on the last event entry in the
        // map is sufficient without waiting for all previous events to
        // complete
        if (!evts.empty() && evts.back()) {
            wait_for_event_completion(evts.back());
        }
    }
    check_for_completed_primitives();

    if (!events_.empty())
        VERROR(primitive, runtime,
                "profiling error: failed to log all pending primitives");

    reset();
}

} // namespace xpu
} // namespace impl
} // namespace dnnl
