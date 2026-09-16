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

#ifndef GPU_GPU_PLUGIN_ABI_HPP
#define GPU_GPU_PLUGIN_ABI_HPP

#include "common/c_types_map.hpp"
#include "common/impl_list_item.hpp"

// This header is the entire surface a GPU plugin shared library needs to
// include in order to register implementations with oneDNN's GPU primitive
// dispatch.
//
// This is an experimental, fork-local extension point -- NOT a stable,
// versioned public API. A plugin must be built against this exact oneDNN
// source tree and compiler; there is no ABI compatibility guarantee across
// oneDNN versions, compilers, or build configurations. Every type crossing
// this boundary by value (impl_list_item_t, and anything a plugin's pd_t/
// primitive_t touches from primitive_desc.hpp/primitive.hpp) must come from
// an identical build of the oneDNN headers on both sides.
//
// A plugin .so must export exactly one symbol, with C linkage so it can be
// found by name via dlsym/find_symbol regardless of the plugin's own C++
// name mangling:
//
//   extern "C" dnnl::impl::status_t dnnl_gpu_plugin_register(
//           dnnl::impl::gpu::plugin_registrar_t &registrar);
//
// That function is called once, at plugin-load time, and should call
// registrar.register_impl(kind, item) once per implementation it wants to
// register (any number of times, for any number of primitive kinds).
// Returning a non-success status only affects logging; it does not undo
// registrations already made in the same call.

namespace dnnl {
namespace impl {
namespace gpu {

// Passed to a plugin's registration entry point. Plugins register any
// number of impl_list_item_t entries against any number of primitive kinds.
// Registered entries are tried, in registration order, ahead of oneDNN's
// own compiled-in implementations for that kind -- see gpu_plugin_registry.hpp.
struct plugin_registrar_t {
    virtual status_t register_impl(
            primitive_kind_t kind, const impl_list_item_t &item)
            = 0;

protected:
    ~plugin_registrar_t() = default;
};

using plugin_register_func_t = status_t (*)(plugin_registrar_t &registrar);

// The symbol name every plugin .so must export with C linkage.
inline const char *plugin_register_symbol_name() {
    return "dnnl_gpu_plugin_register";
}

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
