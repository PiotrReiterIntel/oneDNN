/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "gpu/gpu_plugin_registry.hpp"

#include <cstdlib>
#include <mutex>
#include <string>

#include "common/verbose.hpp"
#include "xpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

namespace {
std::vector<std::string> split_plugin_paths(const char *s) {
    std::vector<std::string> out;
    if (!s) return out;
    std::string str(s);
    size_t pos = 0;
    while (pos <= str.size()) {
        size_t next = str.find(':', pos);
        if (next == std::string::npos) next = str.size();
        if (next > pos) out.emplace_back(str.substr(pos, next - pos));
        pos = next + 1;
    }
    return out;
}
} // namespace

gpu_plugin_registry_t &gpu_plugin_registry_t::instance() {
    static gpu_plugin_registry_t registry;
    static std::once_flag load_once;
    std::call_once(load_once, [] { registry.load_plugins(); });
    return registry;
}

void gpu_plugin_registry_t::load_plugins() {
    const char *env = std::getenv("ONEDNN_GPU_PLUGIN_PATH");
    for (const auto &path : split_plugin_paths(env)) {
        void *sym = xpu::find_symbol(
                path.c_str(), plugin_register_symbol_name());
        if (!sym) {
            // xpu::find_symbol has already logged the failure via VERROR.
            continue;
        }
        auto register_fn
                = reinterpret_cast<plugin_register_func_t>(sym);
        status_t status = register_fn(*this);
        if (status != status::success) {
            VERROR(common, runtime,
                    "gpu plugin '%s' registration returned status %d",
                    path.c_str(), (int)status);
        }
    }
}

status_t gpu_plugin_registry_t::register_impl(
        primitive_kind_t kind, const impl_list_item_t &item) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    registered_[(int)kind].push_back(item);
    merged_.erase((int)kind);
    return status::success;
}

const impl_list_item_t *gpu_plugin_registry_t::get_list(
        primitive_kind_t kind, const impl_list_item_t *builtin) {
    // registered_ is fully populated by load_plugins() before any caller
    // can reach get_list() (both happen behind instance()'s call_once), so
    // this lookup is safe without holding cache_mutex_.
    auto reg_it = registered_.find((int)kind);
    if (reg_it == registered_.end() || reg_it->second.empty()) return builtin;

    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto cached = merged_.find((int)kind);
    if (cached != merged_.end()) return cached->second.data();

    std::vector<impl_list_item_t> merged(reg_it->second);
    for (const impl_list_item_t *p = builtin; p && *p; ++p)
        merged.push_back(*p);
    merged.push_back(impl_list_item_t(nullptr));

    auto ins = merged_.emplace((int)kind, std::move(merged));
    return ins.first->second.data();
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
