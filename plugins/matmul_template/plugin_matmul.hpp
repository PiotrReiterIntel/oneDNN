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

#ifndef PLUGIN_MATMUL_HPP
#define PLUGIN_MATMUL_HPP

// Template matmul GPU plugin: same pd_t/primitive_t shape as
// kf_matmul_plugin/kf_matmul.hpp, with the kernel-specific parts left as
// TODOs. See that file for a filled-in reference.

#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/gpu_matmul_pd.hpp"

namespace plugin {

// DECLARE_COMMON_PD_T (used below) expands to code that references
// oneDNN's common types (status_t, utils::, cache_blob_t, ...) unqualified
// -- including the bare namespace name "impl" itself -- which only
// resolves because ref_matmul_t/etc. live nested directly inside namespace
// dnnl::impl::*, where unqualified lookup walks up to dnnl and finds its
// "impl" member. plugin_matmul_t instead lives in this plugin's own
// top-level namespace, so bring both the unqualified names and the "impl"
// namespace name itself into scope explicitly (see kf_matmul.hpp for the
// same fix).
using namespace dnnl::impl;
namespace impl = dnnl::impl;

struct plugin_matmul_t : public dnnl::impl::gpu::generic::sycl::primitive_t {
    using dnnl::impl::gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public dnnl::impl::gpu::gpu_matmul_pd_t {
        using dnnl::impl::gpu::gpu_matmul_pd_t::gpu_matmul_pd_t;

        // Rename this to identify your kernel in ONEDNN_VERBOSE output.
        DECLARE_COMMON_PD_T("template:matmul:any", plugin_matmul_t);

        dnnl::impl::status_t init(const dnnl::impl::engine_t *engine);

    private:
        dnnl::impl::status_t set_default_params();
        dnnl::impl::status_t scales_ok(
                const dnnl::impl::engine_t *engine) const;

        // This kernel's own accept/reject rule, checked directly against
        // this operation's real oneDNN descriptors/attrs
        // (src_md()/weights_md()/dst_md()/attr()). No translation layer --
        // fill in plugin_matmul.cpp's stub with your kernel's real limits.
        bool kernel_supports() const;
    };

    dnnl::impl::status_t init(dnnl::impl::engine_t *engine) override;
    dnnl::impl::status_t execute(
            const dnnl::impl::exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return (const pd_t *)dnnl::impl::primitive_t::pd().get();
    }

    // TODO: your compiled SYCL kernel handle, set in init() below.
    dnnl::impl::gpu::generic::sycl::kernel_t kernel_;
};

} // namespace plugin

#endif
