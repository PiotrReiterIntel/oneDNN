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

#ifndef KF_MATMUL_HPP
#define KF_MATMUL_HPP

// Example GPU plugin that dispatches straight to KF's own real evolvable
// kernel (kf_kernel.sycl.hpp) instead of a stand-in kernel of oneDNN's
// own. kf_kernel.sycl.hpp is not hand-maintained -- it's mechanically
// regenerated from KF's actual tasks/onednn/matmul_relu/kernel.sycl by
// update_kernel.sh (see that script for what it does and why). Iterating
// on the kernel is: rerun update_kernel.sh against the updated
// kernel.sycl, rebuild this plugin's .so, rerun. There is no create_kernel()/parallel_for() device-functor
// submission here, unlike matmul_template/ref_matmul_t -- KF's kernel is a
// plain host-side function that takes raw USM device pointers and submits
// its own work against a queue, exactly like it does today from its
// PyTorch wrapper. execute() below just hands it that queue (pulled from
// oneDNN's own stream) and the raw USM pointers behind this operation's
// src/weights/dst memory, and gets out of the way.
//
// This only works when oneDNN's GPU memory is USM-backed (benchdnn's
// default, memory_kind_ext_t::usm) -- see execute()'s comment. Buffer-
// backed memory has no raw pointer to hand KF's kernel; that's a real
// constraint of reusing a raw-pointer kernel signature as-is, not
// something this plugin works around.
//
// The accept/reject gate in pd_t::init() (kf_kernel_supports(),
// kf_matmul.cpp) is hardcoded to exactly what KF's tasks/onednn/matmul_relu
// task computes today: plain 2D f32 C = relu(A @ B), no bias, no
// scales/zero-points, fixed (non-runtime) dims. That's the accept/reject
// logic this exercise is about keeping fixed across kernel iterations --
// only kf_kernel.sycl.hpp's evolved body is expected to change.

#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/gpu_matmul_pd.hpp"

namespace kf_plugin {

// DECLARE_COMMON_PD_T (used below) expands to code that references
// oneDNN's common types (status_t, utils::, cache_blob_t, ...) unqualified
// -- including the bare namespace name "impl" itself (as in "impl::
// primitive_t") -- which only resolves because ref_matmul_t/etc. live
// nested directly inside namespace dnnl::impl::*, where unqualified lookup
// walks up to dnnl and finds its "impl" member. kf_matmul_t instead lives
// in this plugin's own top-level namespace, so bring both the unqualified
// names and the "impl" namespace name itself into scope explicitly.
using namespace dnnl::impl;
namespace impl = dnnl::impl;

struct kf_matmul_t : public dnnl::impl::gpu::generic::sycl::primitive_t {
    using dnnl::impl::gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public dnnl::impl::gpu::gpu_matmul_pd_t {
        using dnnl::impl::gpu::gpu_matmul_pd_t::gpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("kf:custom:any", kf_matmul_t);

        dnnl::impl::status_t init(const dnnl::impl::engine_t *engine);

    private:
        // This kernel variant's own accept/reject rule, checked directly
        // against this operation's real oneDNN descriptors/attrs
        // (src_md()/weights_md()/dst_md()/attr()). No translation layer --
        // see kf_matmul.cpp for the actual rule.
        bool kf_kernel_supports() const;

        dnnl::impl::status_t set_default_params();
        dnnl::impl::status_t scales_ok(
                const dnnl::impl::engine_t *engine) const;
    };

    dnnl::impl::status_t init(dnnl::impl::engine_t *engine) override;
    dnnl::impl::status_t execute(
            const dnnl::impl::exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return (const pd_t *)dnnl::impl::primitive_t::pd().get();
    }
};

} // namespace kf_plugin

#endif
