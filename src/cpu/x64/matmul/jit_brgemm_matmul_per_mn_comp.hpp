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

#ifndef CPU_X64_MATMUL_JIT_BRGEMM_MATMUL_PER_MN_COMP_HPP
#define CPU_X64_MATMUL_JIT_BRGEMM_MATMUL_PER_MN_COMP_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

// JIT kernel that fills the per-(M, N) f32 compensation tile consumed by
// the brgemm post-ops epilogue. For one (m_blk, n_blk) tile of one batch
// element it computes
//
//   Δ[m,n] = sum_g s_src[g,m] · s_wei[g,n]
//              · ( z_src[g,m]·S[g,n] + z_wei[g,n]·T[g,m]
//                  − G_g · z_src[g,m] · z_wei[g,n] )
//
// where T[g,m] = sum_{k∈g} src[m,k], S[g,n] = sum_{k∈g} wei[k,n], G_g = |g|.
//
// Construction time (in `brgemm_matmul_t::init`):
//   per_mn_comp_kernel_t::create(kernel, bgmmc)
//     - returns `status::unimplemented` if the host ISA / dtype mix is
//       not supported; the primitive itself then declines and the
//       dispatcher tries the next impl.
//     - on success the kernel object owns its Xbyak code blob.
//
// Per-thread scratch (registered by `init_scratchpad`):
//   per_thread_scratch_bytes(bgmmc) returns the total byte size. The
//   buffer is split into six aligned sub-arrays (T, S, z_src, z_wei,
//   s_src, s_wei); call sites compute the sub-pointers via
//   `scratch_layout_t` populated from the same `bgmmc`.
//
// Per-call (`fill_per_mn_compensation`):
//   ctx_t fills tile pointers (un-shifted batch bases) and the scratch
//   pointers; `operator()` walks K-groups and writes Δ.
struct per_mn_comp_kernel_t {

    struct ctx_t {
        // Un-shifted batch base pointers for src / wei. Tile-origin
        // offsets are passed via `m_base` / `n_base` and applied in
        // ELEMENT units so sub-byte data types extend safely.
        const void *src_batch_ptr = nullptr;
        const void *wei_batch_ptr = nullptr;

        // Batch-adjusted zero-point / scales pointers
        const void *src_zp_ptr = nullptr;
        const void *wei_zp_ptr = nullptr;
        const void *src_scales_ptr = nullptr;
        const void *wei_scales_ptr = nullptr;

        // f32 delta tile output. Layout follows brgemm's buffer_c stripe
        // pattern (see comment on Δ in the file header).
        float *delta_ptr = nullptr;

        // Per-thread scratch (six sub-arrays from a single scratchpad
        // allocation). All six must be valid; pass the corresponding
        // sub-pointers carved out by `scratch_layout_t::pointers_in`.
        int32_t *T_buf = nullptr; // M_blk × i32
        int32_t *S_buf = nullptr; // N_blk × num_k_groups × i32
        int32_t *z_src_buf = nullptr; // M_blk × i32
        int32_t *z_wei_buf = nullptr; // N_blk × i32
        float *s_src_buf = nullptr; // M_blk × f32
        float *s_wei_buf = nullptr; // N_blk × f32

        // Tile coordinates.
        dim_t m_base = 0;
        dim_t n_base = 0;
        dim_t M_blk = 0;
        dim_t N_blk = 0;

        bool s_buf_valid = false;
    };

    // Scratchpad layout helper. The init_scratchpad call sizes a single
    // contiguous per-thread allocation; this helper splits it into the
    // six typed sub-arrays in the call's `ctx_t`.
    struct scratch_layout_t {
        size_t T_off_bytes = 0;
        size_t S_off_bytes = 0;
        size_t z_src_off_bytes = 0;
        size_t z_wei_off_bytes = 0;
        size_t s_src_off_bytes = 0;
        size_t s_wei_off_bytes = 0;
        size_t total_bytes = 0;

        // Compute the layout from a `bgmmc`. The per-thread buffer
        // sizes upper-bound by M_blk / N_blk.
        static scratch_layout_t from_conf(const brgemm_matmul_conf_t &bgmmc);

        // Populate the six ctx pointers from a per-thread scratch base.
        void pointers_in(ctx_t &ctx, char *scratch_base) const;
    };

    // Total per-thread scratch bytes (sum of all six sub-arrays plus
    // alignment slack).
    static size_t per_thread_scratch_bytes(const brgemm_matmul_conf_t &bgmmc) {
        return scratch_layout_t::from_conf(bgmmc).total_bytes;
    }

    // Factory. Returns `status::unimplemented` when the JIT cannot be
    // built for `bgmmc` (unsupported ISA / dtype / stride combo).
    static status_t create(std::unique_ptr<per_mn_comp_kernel_t> &kernel,
            const brgemm_matmul_conf_t *bgmmc);

    virtual void operator()(const ctx_t *ctx) const = 0;
    virtual ~per_mn_comp_kernel_t() = default;

protected:
    per_mn_comp_kernel_t() = default;
};

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
