/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GPU_INTEL_MATMUL_REF_GROUPED_GEMM_HPP
#define GPU_INTEL_MATMUL_REF_GROUPED_GEMM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/intel/matmul/config.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

// Two grouped matmul patterns are supported:
// 2D grouped src (variable M) x 3D dense wei -> 2D grouped dst (variable M)
// 2D grouped src (variable K) x 2D grouped wei (variable M) -> dense 3D dst
struct ref_grouped_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:ref_grouped:any", ref_grouped_t);

        // For 3D weights [G, K, N], override masks to include 0th expert dim
        int wei_qmask_K() const { return (1 << 0) | (1 << 1); }
        int wei_qmask_N() const { return (1 << 0) | (1 << 2); }

        bool is_2dby2d() const { return is_2dby2d_; }

        status_t init(impl::engine_t *engine) {
            src_dt_ = src_md()->data_type;
            dst_dt_ = dst_md()->data_type;
            wei_dt_ = weights_md(0)->data_type;

            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));

            // Detect pattern (2Dx3D vs 2Dx2D) and initialize
            VDISPATCH_MATMUL(
                    src_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            is_2dby2d_ = wei_d.is_grouped_desc();

            const auto &src_grouped = src_d.sparse_desc().grouped_desc;
            group_count_ = src_grouped.group_count;

            return is_2dby2d_ ? init_2dby2d(engine) : init_2dby3d(engine);
        }

        data_type_t src_dt_ = data_type::undef;
        data_type_t dst_dt_ = data_type::undef;
        data_type_t wei_dt_ = data_type::undef;
        dim_t group_count_ = 0;

    private:
        bool is_2dby2d_ = false;

        status_t init_2dby3d(impl::engine_t *engine) {
            using namespace data_type;

            memory_desc_wrapper wei_d(weights_md(0));
            memory_desc_wrapper dst_d(dst_md());

            // Supported configurations: grouped src/dst, dense 3D weights
            VDISPATCH_MATMUL(
                    dst_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(wei_d.is_blocking_desc() && wei_d.ndims() == 3,
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            // GPU ref currently only supports matching data types
            VDISPATCH_MATMUL(src_dt_ == wei_dt_ && src_dt_ == dst_dt_
                            && utils::one_of(src_dt_, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            // Check for supported quantization schemes
            const auto &attr_scales = attr()->scales_;
            if (!attr_scales.has_default_values(DNNL_ARG_SRC)) {
                const int src_mask = attr_scales.get_mask(DNNL_ARG_SRC);
                const int rowwise_mask = src_qmask_M();
                // Only row-wise f32 scales supported for src
                VDISPATCH_MATMUL(src_mask == rowwise_mask,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(attr_scales.get_data_type(DNNL_ARG_SRC) == f32,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                // No groups for src scales
                VDISPATCH_MATMUL(
                        attr_scales.get(DNNL_ARG_SRC).has_default_groups(),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            }
            if (!attr_scales.has_default_values(DNNL_ARG_WEIGHTS)) {
                const int wei_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
                const int colwise_mask = wei_qmask_N();
                // Only column-wise f32 scales supported for weights
                VDISPATCH_MATMUL(wei_mask == colwise_mask,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(
                        attr_scales.get_data_type(DNNL_ARG_WEIGHTS) == f32,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                // No groups for weight scales
                VDISPATCH_MATMUL(
                        attr_scales.get(DNNL_ARG_WEIGHTS).has_default_groups(),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            }
            VDISPATCH_MATMUL(attr_scales.has_default_values(DNNL_ARG_DST),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            // Zero-points are not supported
            VDISPATCH_MATMUL(attr()->zero_points_.has_default_values(),
                    VERBOSE_UNSUPPORTED_ATTR);

            // Only eltwise and binary post-ops are supported
            VDISPATCH_MATMUL(
                    attr()->post_ops_.has_default_values(
                            {primitive_kind::eltwise, primitive_kind::binary}),
                    VERBOSE_UNSUPPORTED_POSTOP);

            // Below is to allow using format_any for scalar [1, 1] as binary
            // post-ops for NVFP4 global scale support
            const auto &po = attr()->post_ops_;
            for (int i = 0; i < po.len(); ++i) {
                auto &e = attr_.post_ops_.entry_[i];
                if (e.is_binary()) {
                    const memory_desc_wrapper src1_d(e.binary.src1_desc);
                    if (src1_d.format_any()) {
                        VDISPATCH_MATMUL(src1_d.count_non_unit_dims(0),
                                VERBOSE_UNSUPPORTED_POSTOP);
                        CHECK(memory_desc_init_by_strides(
                                e.binary.src1_desc, nullptr));
                    }
                }
            }

            return status::success;
        }

        status_t init_2dby2d(impl::engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_MATMUL(dst_d.is_blocking_desc() && dst_d.ndims() == 3,
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            VDISPATCH_MATMUL(src_type == wei_type && src_type == dst_type
                            && utils::one_of(src_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_MATMUL(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(!with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

            return status::success;
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.set_data_type(pd()->dst_dt_);
        def_data_type(kernel_ctx, pd()->src_dt_, "SRC");
        def_data_type(kernel_ctx, pd()->wei_dt_, "WEI");
        def_data_type(kernel_ctx, pd()->dst_dt_, "DST");
        def_data_type(kernel_ctx, pd()->desc()->accum_data_type, "ACC");

        auto attr_info = attr_info_t::create(pd()->attr());
        CHECK(def_attr_info(kernel_ctx, attr_info, pd()->attr()->post_ops_,
                *pd()->dst_md()));

        const bool with_bias = pd()->with_bias();
        kernel_ctx.define_int("WITH_BIAS", with_bias ? 1 : 0);
        if (with_bias)
            def_data_type(kernel_ctx, pd()->weights_md(1)->data_type, "BIA");

        return create_kernel(
                engine, &kernel_, "ref_grouped_gemm_matmul", kernel_ctx);
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
#endif // GPU_INTEL_MATMUL_REF_GROUPED_GEMM_HPP
