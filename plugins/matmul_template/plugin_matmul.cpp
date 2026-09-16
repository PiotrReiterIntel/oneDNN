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

#include "plugin_matmul.hpp"

#include "common/impl_list_item.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_plugin_abi.hpp"

// Convenience -- this translation unit is the plugin's own, not part of
// oneDNN's namespace tree, so pull in the names ref_matmul.cpp gets for
// free by living inside dnnl::impl::gpu::generic::sycl.
using namespace dnnl::impl;
using namespace dnnl::impl::gpu::generic::sycl;

namespace plugin {

namespace {

// Implementation details of pd_t::init() below -- static, no pd_t state,
// so no reason for these to be class members.
bool check_data_types(const memory_desc_wrapper &src,
        const memory_desc_wrapper &weights, const memory_desc_wrapper &dst) {
    using namespace data_type;

    // Generic oneDNN-supported matmul dtypes. Your capability check
    // (kernel_supports, below) is what actually narrows this down to
    // what your kernel handles -- this is only the outer bound.
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

bool plugin_matmul_t::pd_t::kernel_supports() const {
    // TODO: replace with your kernel's real support rule. Returning
    // false always is a safe default -- every shape falls through to
    // oneDNN's own matmul implementations until this is filled in. You
    // have full access to this operation's real oneDNN descriptors and
    // attrs here (src_md()/weights_md()/dst_md()/attr()) -- no
    // translation layer required.
    return false;
}

status_t plugin_matmul_t::pd_t::set_default_params() {
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

status_t plugin_matmul_t::pd_t::scales_ok(const engine_t *engine) const {
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

status_t plugin_matmul_t::pd_t::init(const engine_t *engine) {
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
    VDISPATCH_MATMUL(sycl_post_ops_t::post_ops_ok(attr()),
            VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_MATMUL(md_dims_in_range(src_md()), VERBOSE_OUT_OF_RANGE_DIMS,
            "src");
    VDISPATCH_MATMUL(md_dims_in_range(weights_md()),
            VERBOSE_OUT_OF_RANGE_DIMS, "weights");
    VDISPATCH_MATMUL(
            IMPLICATION(!attr()->zero_points_.has_default_values(),
                    !attr()->zero_points_.has_host_scalars()),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    // The one call that ties this pd_t to your kernel's real limits --
    // everything above is generic oneDNN-shape validation any matmul
    // implementation needs. Keeping this check right next to the kernel
    // it gates (instead of in a separately-maintained validator) is the
    // whole point of this plugin mechanism.
    VDISPATCH_MATMUL(kernel_supports(), VERBOSE_UNSUPPORTED_DT_CFG);

    // TODO: finalize whatever config your kernel needs, and book
    // scratchpad via scratchpad_registry() if it needs one. See
    // kf_matmul_plugin/kf_matmul.cpp's init_conf()/init_rt_conf() for a
    // worked example of precomputing kernel launch parameters here.

    return status::success;
}

status_t plugin_matmul_t::init(engine_t *engine) {
    // TODO: build/compile your SYCL kernel, e.g.:
    //   const auto kid = ::sycl::get_kernel_id<your_kernel_t>();
    //   CHECK(create_kernel(engine, kid, &kernel_));
    return status::unimplemented;
}

status_t plugin_matmul_t::execute(const exec_ctx_t &ctx) const {
    // TODO: submit kernel_ via parallel_for(ctx, ...). See
    // kf_matmul_plugin/kf_matmul.cpp's execute() for a worked example.
    return status::unimplemented;
}

// Plugin ABI entry point -- see gpu_plugin_abi.hpp. Exported with C
// linkage so gpu_plugin_registry_t can find it by name via dlsym
// regardless of this .so's own C++ name mangling. Visibility is forced
// to "default" since oneDNN's own build flags (which this plugin's
// CMakeLists reuses) set -fvisibility=internal for everything by
// default -- without this attribute the symbol would be compiled but
// invisible to dlsym.
extern "C" __attribute__((visibility("default"))) dnnl::impl::status_t
dnnl_gpu_plugin_register(dnnl::impl::gpu::plugin_registrar_t &registrar) {
    using namespace dnnl::impl;
    impl_list_item_t item {impl_list_item_t::type_deduction_helper_t<
            plugin_matmul_t::pd_t>()};
    return registrar.register_impl(primitive_kind::matmul, item);
}

} // namespace plugin
