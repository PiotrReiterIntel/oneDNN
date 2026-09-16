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

#include "kf_matmul.hpp"

#include "common/impl_list_item.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/generic/sycl/stream.hpp"
#include "gpu/gpu_plugin_abi.hpp"
#include "xpu/sycl/memory_storage_base.hpp"

#include "kf_kernel.sycl.hpp"

// Convenience -- this translation unit is the plugin's own, not part of
// oneDNN's namespace tree, so pull in the names ref_matmul.cpp gets for
// free by living inside dnnl::impl::gpu::generic::sycl.
using namespace dnnl::impl;
using namespace dnnl::impl::gpu::generic::sycl;

namespace kf_plugin {

namespace {

// Implementation details of pd_t::init() below -- static, no pd_t state,
// so no reason for these to be class members.
bool check_data_types(const memory_desc_wrapper &src,
        const memory_desc_wrapper &weights, const memory_desc_wrapper &dst) {
    using namespace data_type;

    for (auto t : {src.data_type(), weights.data_type(), dst.data_type()}) {
        if (!utils::one_of(t, f32, bf16, f16, s8, u8, s32)) return false;
    }
    return true;
}

bool check_formats(const memory_desc_wrapper &src,
        const memory_desc_wrapper &weights, const memory_desc_wrapper &dst) {
    for (const auto &mdw : {src, weights, dst}) {
        if (!mdw.is_plain()) return false;
    }
    return true;
}

} // namespace

bool kf_matmul_t::pd_t::kf_kernel_supports() const {
    // Hardcoded to exactly what kf_kernel.sycl.hpp's evolved sycl_kernel()
    // computes today: plain 2D f32 C = relu(A @ B), no bias, no
    // scales/zero-points, fixed (non-runtime) dims. This is the fixed
    // accept/reject rule this exercise keeps stable across kernel
    // iterations -- only kf_kernel.sycl.hpp's evolved body is meant to
    // change as the kernel improves. Anything outside this one shape
    // falls through to oneDNN's own matmul implementations.
    using namespace data_type;

    if (ndims() != 2) return false;
    if (with_bias()) return false;

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper weights_d(weights_md(0));
    const memory_desc_wrapper dst_d(dst_md());

    if (src_d.data_type() != f32 || weights_d.data_type() != f32
            || dst_d.data_type() != f32)
        return false;
    if (src_d.has_runtime_dims_or_strides()
            || weights_d.has_runtime_dims_or_strides()
            || dst_d.has_runtime_dims_or_strides())
        return false;

    const auto &po = attr()->post_ops_;
    if (po.len() != 1 || !po.entry_[0].is_relu(true, true)) return false;

    return true;
}

status_t kf_matmul_t::pd_t::set_default_params() {
    if (src_md_.format_kind == format_kind::any) {
        auto src_tag = utils::pick(
                ndims() - 2, format_tag::ab, format_tag::abc,
                format_tag::abcd);
        CHECK(memory_desc_init_by_tag(src_md_, src_tag));
    }
    const memory_desc_wrapper src_d(src_md());
    if (src_d.is_blocking_desc()) {
        if (weights_md_.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_blocking_desc(
                    weights_md_, src_d.blocking_desc()));
        }
        if (dst_md_.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_blocking_desc(
                    dst_md_, src_d.blocking_desc()));
        }
    }
    const memory_desc_wrapper dst_d(dst_md());
    if (dst_d.is_blocking_desc()) {
        if (bias_md_.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_blocking_desc(
                    bias_md_, dst_d.blocking_desc()));
        }
    }
    return status::success;
}

status_t kf_matmul_t::pd_t::scales_ok(const engine_t *engine) const {
    const std::vector<int> supported_args
            = {DNNL_ARG_SRC_0, DNNL_ARG_WEIGHTS_0, DNNL_ARG_DST};

    const auto &scales = attr()->scales_;
    for (auto arg : supported_args) {
        if (!scales.get(arg).has_default_values()) {
            VDISPATCH_MATMUL(is_supported_type(scales.get_data_type(arg)),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_MATMUL(!scales.get(arg).is_host_scalar(),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }
    }
    CHECK(attr_scales_ok(engine, supported_args));
    return status::success;
}

status_t kf_matmul_t::pd_t::init(const engine_t *engine) {
    using namespace data_type;
    using sm = primitive_attr_t::skip_mask_t;

    VDISPATCH_MATMUL_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL_SC(attr_.set_default_formats(dst_md()),
            VERBOSE_UNSUPPORTED_ATTR);

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper weights_d(weights_md(0));
    const memory_desc_wrapper dst_d(dst_md());

    VDISPATCH_MATMUL(check_data_types(src_d, weights_d, dst_d),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(check_formats(src_d, weights_d, dst_d),
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(
            attr()->has_default_values(sm::post_ops | sm::dropout
                    | sm::scales_data_type | sm::zero_points_data_type),
            VERBOSE_UNSUPPORTED_ATTR);
    CHECK(scales_ok(engine));
    VDISPATCH_MATMUL(md_dims_in_range(src_md()), VERBOSE_OUT_OF_RANGE_DIMS,
            "src");
    VDISPATCH_MATMUL(md_dims_in_range(weights_md()),
            VERBOSE_OUT_OF_RANGE_DIMS, "weights");
    VDISPATCH_MATMUL(
            IMPLICATION(!attr()->zero_points_.has_default_values(),
                    !attr()->zero_points_.has_host_scalars()),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    // The plugin's actual accept/reject decision, kept immediately next to
    // the kernel it gates rather than in a separately-maintained validator
    // (the exact flaw this plugin mechanism exists to fix). Everything
    // above is generic oneDNN-shape validation any matmul implementation
    // needs; this is what makes THIS kernel variant say yes or no.
    VDISPATCH_MATMUL(kf_kernel_supports(), VERBOSE_UNSUPPORTED_DT_CFG);

    return status::success;
}

status_t kf_matmul_t::init(engine_t *engine) {
    return status::success;
}

status_t kf_matmul_t::execute(const exec_ctx_t &ctx) const {
    const memory_desc_wrapper dst_d(pd()->dst_md());
    if (dst_d.size() == 0) return status::success;

    const memory_desc_wrapper src_d(pd()->src_md());

    // KF's kernel is a plain host-side function that submits its own SYCL
    // work against a queue we hand it, using raw USM device pointers --
    // exactly like it does today from its PyTorch wrapper (see
    // kf_kernel.sycl.hpp). No create_kernel()/parallel_for() device-
    // functor submission is involved on this path.
    auto *stream = utils::downcast<gpu::generic::sycl::stream_t *>(
            ctx.stream());
    ::sycl::queue &queue = stream->queue();

    auto get_usm_ptr = [](const memory_storage_t &storage) -> void * {
        // This plugin only works against USM-backed memory (benchdnn's
        // default: memory_kind_ext_t::usm) -- there is no raw device
        // pointer to hand a raw-pointer kernel signature when memory is
        // buffer-backed instead. Fail loudly rather than silently reading
        // a garbage handle.
        assert(utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                       &storage)
                       ->memory_kind()
                == xpu::sycl::memory_kind::usm);
        return storage.data_handle();
    };

    const auto *A = static_cast<const float *>(
            get_usm_ptr(CTX_IN_STORAGE(DNNL_ARG_SRC)));
    const auto *B = static_cast<const float *>(
            get_usm_ptr(CTX_IN_STORAGE(DNNL_ARG_WEIGHTS)));
    auto *C = static_cast<float *>(
            get_usm_ptr(CTX_OUT_STORAGE(DNNL_ARG_DST)));

    const int M = static_cast<int>(src_d.dims()[0]);
    const int K = static_cast<int>(src_d.dims()[1]);
    const int N = static_cast<int>(dst_d.dims()[1]);

    sycl_kernel(queue, A, B, C, M, K, N);

    return status::success;
}

// Plugin ABI entry point -- see gpu_plugin_abi.hpp. Exported with C
// linkage so gpu_plugin_registry_t can find it by name via dlsym
// regardless of this .so's own C++ name mangling. Visibility is forced to
// "default" since oneDNN's own build flags (which this plugin's CMakeLists
// reuses) set -fvisibility=internal for everything by default -- without
// this attribute the symbol would be compiled but invisible to dlsym.
extern "C" __attribute__((visibility("default"))) dnnl::impl::status_t
dnnl_gpu_plugin_register(dnnl::impl::gpu::plugin_registrar_t &registrar) {
    using namespace dnnl::impl;
    impl_list_item_t item{
            impl_list_item_t::type_deduction_helper_t<kf_matmul_t::pd_t>()};
    return registrar.register_impl(primitive_kind::matmul, item);
}

} // namespace kf_plugin
