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

#ifndef GPU_GPU_PLUGIN_REGISTRY_HPP
#define GPU_GPU_PLUGIN_REGISTRY_HPP

#include <map>
#include <mutex>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/impl_list_item.hpp"
#include "gpu/gpu_plugin_abi.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

// Lazily loads GPU plugin shared libraries named in the ':'-separated
// ONEDNN_GPU_PLUGIN_PATH environment variable (Linux only), and merges their
// registered implementations ahead of oneDNN's own compiled-in impl list for
// the corresponding primitive kind. See gpu_plugin_abi.hpp for what a plugin
// must export.
//
// This is an experimental, fork-local extension point. Plugins are loaded
// once, at first use, and never dlclose'd -- see xpu::find_symbol, which
// this class uses to load plugin libraries.
class gpu_plugin_registry_t : public plugin_registrar_t {
public:
    static gpu_plugin_registry_t &instance();

    // plugin_registrar_t
    status_t register_impl(
            primitive_kind_t kind, const impl_list_item_t &item) override;

    // Returns a null-terminated impl_list_item_t array consisting of every
    // plugin-registered entry for `kind` (in registration order), followed
    // by `builtin`. Returns `builtin` unchanged when no plugin has
    // registered anything for `kind` -- no allocation, no locking beyond a
    // single map lookup.
    const impl_list_item_t *get_list(
            primitive_kind_t kind, const impl_list_item_t *builtin);

private:
    gpu_plugin_registry_t() = default;
    void load_plugins();

    // Populated once, during load_plugins(); read-only afterwards, so
    // get_list() can consult it without locking.
    std::map<int, std::vector<impl_list_item_t>> registered_;

    // Lazily-built merged arrays, cached per primitive_kind_t on first
    // request. Guarded by cache_mutex_ since concurrent primitive creation
    // may race to build the cache entry for the same kind.
    std::mutex cache_mutex_;
    std::map<int, std::vector<impl_list_item_t>> merged_;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
