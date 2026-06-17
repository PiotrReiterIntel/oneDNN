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

#include "cpu/x64/matmul/jit_brgemm_matmul_per_mn_comp.hpp"

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"
#include "cpu/ref_io_helper.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace Xbyak;

namespace {

// Round-up to alignment.
inline size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

// -----------------------------------------------------------------------------
// JIT generator: one K-group of the per-(M, N) compensation.
//
// Templated on Vmm (Xbyak::Zmm for AVX-512, Xbyak::Ymm for AVX2/AVX2-
// VNNI). The ISA dispatch is driven by `bgmmc->isa` via the factory.
//
// For one K-group:
//   * (optionally, on first_group) zero the delta tile.
//   * If has_wei_zp_: T_buf[m] = sum_{k ∈ [k_start, k_end)} src[m_base+m, k]
//   * If has_src_zp_: S_buf[n] = sum_{k ∈ [k_start, k_end)} wei[k, n_base+n]
//   * Combine: delta[m,n] += s_src[m] · s_wei[n] · float(
//                                z_src[m]·S[n] + z_wei[n]·T[m]
//                                − G · z_src[m] · z_wei[n] )
//
// Two static modes baked into the kernel at construction:
//   * src_is_signed_, wei_is_signed_  (s8 vs u8; s4 vs u4 for wei)
//   * has_src_zp_, has_wei_zp_        (gates each sub-term)
//   * src_m_stride_bytes_ / src_k_stride_bytes_ (src layout)
//   * wei_k_stride_bytes_ / wei_n_stride_bytes_ (wei layout)
//   * LDC_, M_blk_static_             (stripe walk for delta tile)
//
// Per-call params (call_params_t) are runtime pointers + tile dims +
// k_start/k_end + first_group flag. The group length G is materialized
// at preamble as `reg_k_len = k_end - k_start` and reused.
// -----------------------------------------------------------------------------
template <typename Vmm>
struct jit_per_mn_comp_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_per_mn_comp_kernel_t)

    struct call_params_t {
        const void *src_batch_ptr;
        const void *wei_batch_ptr;
        int32_t *T_buf;
        int32_t *S_buf;
        const int32_t *z_src_buf;
        const int32_t *z_wei_buf;
        const float *s_src_buf;
        const float *s_wei_buf;
        float *delta_ptr;
        dim_t m_base;
        dim_t n_base;
        dim_t M_blk;
        dim_t N_blk;
        dim_t k_start;
        dim_t k_end;
        // 0 on first K-group (zero the delta tile before accumulating),
        // 1 on subsequent groups. Declared dim_t so that the 64-bit
        // load_param reads a known (non-garbage) value.
        dim_t first_group;
        dim_t skip_s_reduce;
    };

    jit_per_mn_comp_kernel_t(const brgemm_matmul_conf_t *bgmmc)
        : jit_generator_t(jit_name())
        , isa_(bgmmc->isa)
        , src_is_signed_(bgmmc->orig_src_dt == data_type::s8)
        , wei_is_signed_(bgmmc->orig_wei_dt == data_type::s8
                  || bgmmc->orig_wei_dt == data_type::s4)
        , wei_is_subbyte_(utils::one_of(
                  bgmmc->orig_wei_dt, data_type::s4, data_type::u4))
        , has_src_zp_(bgmmc->has_zero_point_a)
        , has_wei_zp_(bgmmc->has_zero_point_b)
        , src_m_stride_bytes_(bgmmc->A_strides[1])
        , src_k_stride_bytes_(bgmmc->A_strides[0])
        , wei_k_stride_bytes_(wei_is_subbyte_ ? bgmmc->B_strides[1] / 2
                                              : bgmmc->B_strides[1])
        , LDC_(bgmmc->LDC)
        , M_blk_static_(bgmmc->M_blk) {}

private:
    using reg64_t = const Xbyak::Reg64;
    using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;

#define GET_OFF(x) offsetof(call_params_t, x)

    const cpu_isa_t isa_;
    const bool src_is_signed_;
    const bool wei_is_signed_;
    const bool wei_is_subbyte_;
    const bool has_src_zp_;
    const bool has_wei_zp_;
    const dim_t src_m_stride_bytes_;
    const dim_t src_k_stride_bytes_;
    const dim_t wei_k_stride_bytes_;
    const dim_t LDC_;
    const dim_t M_blk_static_;

    static constexpr int f32_sz = sizeof(float);
    static constexpr int i32_sz = sizeof(int32_t);
    // Dword lanes per Vmm: 16 for Zmm, 8 for Ymm.
    static constexpr int n_lanes = vreg_traits_t<Vmm>::vlen / i32_sz;
    // Bytes per Vmm: 64 for Zmm, 32 for Ymm.
    static constexpr int vmm_bytes = vreg_traits_t<Vmm>::vlen;
    static constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

    // -------------------------------------------------------------------------
    // Register map
    // -------------------------------------------------------------------------
    // ABI param ptr (live throughout — re-read of dim_t fields).
    reg64_t reg_param = abi_param1;

    // Persistent runtime values loaded once at preamble.
    reg64_t reg_M_blk = r12;
    reg64_t reg_N_blk = r13;
    reg64_t reg_k_len = r14;

    // General-purpose scratch (loop counters, transient bases).
    reg64_t reg_row_ptr = r8;
    reg64_t reg_n_chunk_ptr = r9;
    reg64_t reg_iter = r10;
    reg64_t reg_k_iter = r11;
    reg64_t reg_n = rax;
    reg64_t reg_m = rdx;
    reg64_t reg_delta_base = r15;
    reg64_t reg_delta_row = rbx;
    reg64_t reg_tmp = rsi;

    Opmask k_mask = k1;
    // Secondary mask used by the sub-byte S-reduce path: gates the
    // byte-load (one bit per packed byte) while `k_mask` gates the
    // unpacked element store (one bit per N element).
    Opmask k_byte_mask = k2;

    // Vmm allocation (indices are the same for Zmm and Ymm).
    Vmm zmm_ones = Vmm(30); // 0x01 broadcast for VPDPBUSD
    Vmm zmm_zero = Vmm(31); // 0 broadcast for delta reset

    // Reduce scratch (overlaps with combine scratch — phases are disjoint).
    Vmm zmm_acc = Vmm(0);
    Vmm zmm_data = Vmm(1);
    Vmm zmm_hreduce = Vmm(2);

    // Combine scratch (per-m broadcasts + per-n-chunk vectors).
    Vmm zmm_ss_bc = Vmm(3);
    Vmm zmm_zs_bc = Vmm(4);
    Vmm zmm_T_bc = Vmm(5);
    Vmm zmm_Gzs_bc = Vmm(6);
    Vmm zmm_sw = Vmm(7);
    Vmm zmm_zw = Vmm(8);
    Vmm zmm_S = Vmm(9);
    Vmm zmm_t1 = Vmm(10);
    Vmm zmm_t2 = Vmm(11);
    Vmm zmm_t3 = Vmm(12);
    Vmm zmm_term = Vmm(13);
    Vmm zmm_term_f = Vmm(14);
    Vmm zmm_ss_sw = Vmm(15);
    Vmm zmm_delta = Vmm(16);

    // Sub-byte (s4/u4) S-reduce scratch (live only inside
    // emit_s_reduce_subbyte, which never overlaps emit_combine).
    Vmm zmm_perm = Vmm(17); // dword permute table for nibble interleave
    Vmm zmm_0f = Vmm(18); // 0x0000000F broadcast for nibble mask

    // AVX2-only auxiliary Vmms (unused on AVX-512):
    //   vmm_idx_       : {0, 1, ..., n_lanes-1} for tail-mask construction
    //   vmm_tail_mask_ : current tail mask (lanes < count = -1, else 0)
    //   vmm_all_ones_  : all-ones mask (for full chunks). Lives in 29.
    Vmm vmm_idx_ = Vmm(28);
    Vmm vmm_tail_mask_ = Vmm(29);

    void load_param(reg64_t r, int off) { mov(r, ptr[reg_param + off]); }

    Xbyak::Label idx_table_;

    // Set the active mask to cover all n_lanes lanes (full chunk).
    void set_full_mask() {
        if (isa_has_masks(isa_)) {
            mov(reg_tmp, n_lanes == 16 ? 0xFFFF : 0xFF);
            kmovw(k_mask, reg_tmp.cvt32());
        } else {
            // All-ones Vmm via vpcmpeqd self.
            vpcmpeqd(vmm_tail_mask_, vmm_tail_mask_, vmm_tail_mask_);
        }
    }

    void build_low_bits_opmask(Opmask kdst, reg64_t count_reg) {
        assert(isa_has_masks(isa_));
        if (count_reg.getIdx() == reg_tmp.getIdx()) {
            push(rax);
            mov(rax, count_reg);
            mov(reg_tmp, -1);
            bzhi(reg_tmp, reg_tmp, rax);
            pop(rax);
            kmovq(kdst, reg_tmp);
        } else {
            mov(reg_tmp, -1);
            bzhi(reg_tmp, reg_tmp, count_reg);
            kmovq(kdst, reg_tmp);
        }
    }

    void build_low_bits_mask(reg64_t count_reg) {
        if (isa_has_masks(isa_)) {
            build_low_bits_opmask(k_mask, count_reg);
        } else {
            // AVX2: vmm_tail_mask_[i] = (count > i) ? -1 : 0.
            vmovd(Xbyak::Xmm(vmm_tail_mask_.getIdx()), count_reg.cvt32());
            vpbroadcastd(vmm_tail_mask_, Xbyak::Xmm(vmm_tail_mask_.getIdx()));
            vpcmpgtd(vmm_tail_mask_, vmm_tail_mask_, vmm_idx_);
        }
    }

    // Masked f32 load: dst | k_mask | T_z on AVX-512, vmaskmovps on AVX2.
    void masked_load_f32(Vmm dst, const Xbyak::Address &addr) {
        if (isa_has_masks(isa_))
            vmovups(dst | k_mask | T_z, addr);
        else
            vmaskmovps(dst, vmm_tail_mask_, addr);
    }

    // Masked f32 store: addr | k_mask on AVX-512, vmaskmovps on AVX2.
    void masked_store_f32(const Xbyak::Address &addr, Vmm src) {
        if (isa_has_masks(isa_))
            vmovups(addr | k_mask, src);
        else
            vmaskmovps(addr, vmm_tail_mask_, src);
    }

    // Masked i32 load (dword granularity).
    void masked_load_i32(Vmm dst, const Xbyak::Address &addr) {
        if (isa_has_masks(isa_))
            vmovdqu32(dst | k_mask | T_z, addr);
        else
            vpmaskmovd(dst, vmm_tail_mask_, addr);
    }

    // Masked i32 store (dword granularity).
    void masked_store_i32(const Xbyak::Address &addr, Vmm src) {
        if (isa_has_masks(isa_))
            vmovdqu32(addr | k_mask, src);
        else
            vpmaskmovd(addr, vmm_tail_mask_, src);
    }

    // VPDPBUSD with a 0x01 ones vector to fold 4 K-bytes per int32 lane.
    // Encoding follows isa_: EVEX on AVX-512 (vpdpbusd-Zmm), VEX on
    // AVX2-VNNI (vpdpbusd-Ymm). Note: the host CPU may support both, but
    // we always emit the variant matching the dispatched ISA.
    void vpdpbusd_ones(Vmm acc, Vmm data, bool data_is_signed) {
        const auto enc = isa_has_masks(isa_) ? Xbyak::EvexEncoding
                                             : Xbyak::VexEncoding;
        if (data_is_signed)
            vpdpbusd(acc, zmm_ones, data, enc);
        else
            vpdpbusd(acc, data, zmm_ones, enc);
    }

    // Horizontally reduce all dword lanes of `src` into lane 0. `tmp` is
    // a scratch Vmm (clobbered). Handles both Zmm (16 lanes, 5 stages)
    // and Ymm (8 lanes, 4 stages) via compile-time branch.
    void hreduce_vmm_dword(Vmm src, Vmm tmp) {
        const Xbyak::Ymm src_y(src.getIdx());
        const Xbyak::Ymm tmp_y(tmp.getIdx());
        const Xbyak::Xmm src_x(src.getIdx());
        const Xbyak::Xmm tmp_x(tmp.getIdx());
        if (is_zmm) {
            vextracti64x4(tmp_y, Xbyak::Zmm(src.getIdx()), 1);
            vpaddd(src_y, src_y, tmp_y);
        }
        vextracti128(tmp_x, src_y, 1);
        vpaddd(src_x, src_x, tmp_x);
        vpshufd(tmp_x, src_x, 0x4E);
        vpaddd(src_x, src_x, tmp_x);
        vpshufd(tmp_x, src_x, 0xB1);
        vpaddd(src_x, src_x, tmp_x);
    }

    // -------------------------------------------------------------------------
    // Delta tile reset (executed only on first K-group). The reset
    // honors the AMX-style stripe layout: each stripe is M_blk × LDC
    // floats, separated by M_blk * LDC f32 positions. Stripe count is
    // ceil(N_blk / LDC). The last stripe is partial (touches only
    // (N_blk - n_lo) columns per row).
    // -------------------------------------------------------------------------
    void emit_delta_reset() {
        uni_vpxor(zmm_zero, zmm_zero, zmm_zero);

        load_param(reg_delta_base, GET_OFF(delta_ptr));
        xor_(reg_n, reg_n); // global N offset of the current stripe

        Label stripe_loop, stripe_done;
        L(stripe_loop);
        cmp(reg_n, reg_N_blk);
        jge(stripe_done, T_NEAR);

        // Live columns in this stripe = min(LDC, N_blk - n).
        mov(reg_tmp, reg_N_blk);
        sub(reg_tmp, reg_n);
        Label cols_set;
        cmp(reg_tmp, LDC_);
        jle(cols_set, T_NEAR);
        mov(reg_tmp, LDC_);
        L(cols_set);
        // reg_tmp = cols in this stripe.

        // Per-m row, zero `cols` floats. Chunk by n_lanes with a tail
        // mask on the last chunk. We compute the live-cols mask once
        // and reuse it for every m in this stripe — but the full-n_lanes
        // chunks also need a full mask, so just walk one mask per chunk.
        xor_(reg_m, reg_m);
        Label m_loop, m_done;
        L(m_loop);
        cmp(reg_m, reg_M_blk);
        jge(m_done, T_NEAR);

        // delta_row = delta_base + m * LDC * f32
        mov(reg_delta_row, reg_m);
        imul(reg_delta_row, reg_delta_row, LDC_ * f32_sz);
        add(reg_delta_row, reg_delta_base);

        // Walk n_lanes-lane chunks in `cols`.
        xor_(reg_iter, reg_iter); // in-stripe column offset
        Label col_loop, col_done;
        L(col_loop);
        cmp(reg_iter, reg_tmp);
        jge(col_done, T_NEAR);

        // remaining = cols - iter
        push(reg_tmp);
        sub(reg_tmp, reg_iter);
        Label full_mask, mask_set;
        cmp(reg_tmp, n_lanes);
        jge(full_mask, T_NEAR);
        build_low_bits_mask(reg_tmp);
        jmp(mask_set, T_NEAR);
        L(full_mask);
        set_full_mask();
        L(mask_set);
        pop(reg_tmp);

        masked_store_f32(ptr[reg_delta_row + reg_iter * f32_sz], zmm_zero);
        add(reg_iter, n_lanes);
        jmp(col_loop, T_NEAR);
        L(col_done);

        inc(reg_m);
        jmp(m_loop, T_NEAR);
        L(m_done);

        // Advance to next stripe.
        add(reg_n, LDC_);
        add(reg_delta_base, M_blk_static_ * LDC_ * f32_sz);
        jmp(stripe_loop, T_NEAR);
        L(stripe_done);
    }

    // -------------------------------------------------------------------------
    // T-reduce: T_buf[m] = sum_{k ∈ [k_start, k_end)} src[m_base+m, k].
    //
    // For each m, walks the K row in 64-byte chunks (VPDPBUSD with a
    // 0x01 vector folds 4 K-bytes per dword lane), then a masked tail.
    // Final horizontal-sum reduces to one int32 and stores to T_buf[m].
    // -------------------------------------------------------------------------
    void emit_t_reduce() {
        // Broadcast 0x01 byte into all bytes of zmm_ones. The GPR8 form
        // of vpbroadcastb is AVX-512-only; on AVX2 hop through an XMM.
        mov(reg_tmp, 1);
        if (isa_has_masks(isa_)) {
            vpbroadcastb(zmm_ones, reg_tmp.cvt8());
        } else {
            const Xbyak::Xmm ones_x(zmm_ones.getIdx());
            vmovd(ones_x, reg_tmp.cvt32());
            vpbroadcastb(zmm_ones, ones_x);
        }

        // Row base = src_batch_ptr + m_base*src_m_stride + k_start*src_k_stride
        load_param(reg_row_ptr, GET_OFF(src_batch_ptr));
        load_param(reg_tmp, GET_OFF(m_base));
        imul(reg_tmp, reg_tmp, src_m_stride_bytes_);
        add(reg_row_ptr, reg_tmp);
        load_param(reg_tmp, GET_OFF(k_start));
        imul(reg_tmp, reg_tmp, src_k_stride_bytes_);
        add(reg_row_ptr, reg_tmp);

        xor_(reg_m, reg_m);
        Label m_loop, m_done;
        L(m_loop);
        cmp(reg_m, reg_M_blk);
        jge(m_done, T_NEAR);

        uni_vpxor(zmm_acc, zmm_acc, zmm_acc);
        // K-bulk: vmm_bytes per step (64 for Zmm, 32 for Ymm).
        mov(reg_tmp, reg_row_ptr);
        mov(reg_k_iter, reg_k_len);

        Label k_bulk, k_bulk_done;
        L(k_bulk);
        cmp(reg_k_iter, vmm_bytes);
        jl(k_bulk_done, T_NEAR);
        uni_vmovdqu(zmm_data, ptr[reg_tmp]);
        vpdpbusd_ones(zmm_acc, zmm_data, src_is_signed_);
        add(reg_tmp, vmm_bytes);
        sub(reg_k_iter, vmm_bytes);
        jmp(k_bulk, T_NEAR);
        L(k_bulk_done);

        // K-tail: < vmm_bytes bytes remain. AVX-512 uses a masked
        // vmovdqu8 + vpdpbusd; AVX2 has no byte-masked load, so we
        // fall back to a scalar byte-accumulate loop. The tail count
        // is bounded by vmm_bytes-1 (≤63 for Zmm, ≤31 for Ymm), so the
        // scalar loop is negligible vs the bulk VPDPBUSD pass.
        Label k_tail_done;
        test(reg_k_iter, reg_k_iter);
        jz(k_tail_done, T_NEAR);
        if (isa_has_masks(isa_)) {
            // build_low_bits_mask clobbers reg_tmp; reg_tmp here holds
            // the running row pointer, so save/restore it.
            push(reg_tmp);
            build_low_bits_mask(reg_k_iter);
            pop(reg_tmp);
            vmovdqu8(
                    Xbyak::Zmm(zmm_data.getIdx()) | k_mask | T_z, ptr[reg_tmp]);
            vpdpbusd_ones(zmm_acc, zmm_data, src_is_signed_);
        } else {
            // Scalar accumulate: add each tail byte (sign- or zero-
            // extended to s32) to lane 0 of zmm_acc. reg_tmp keeps the
            // row pointer; reg_n_chunk_ptr is the byte-load scratch
            // (unused elsewhere in t_reduce); reg_n is the byte index.
            const Xbyak::Xmm acc_x(zmm_acc.getIdx());
            xor_(reg_n, reg_n);
            Label scalar_loop;
            L(scalar_loop);
            if (src_is_signed_)
                movsx(reg_n_chunk_ptr.cvt32(), byte[reg_tmp + reg_n]);
            else
                movzx(reg_n_chunk_ptr.cvt32(), byte[reg_tmp + reg_n]);
            vmovd(Xbyak::Xmm(zmm_hreduce.getIdx()), reg_n_chunk_ptr.cvt32());
            vpaddd(acc_x, acc_x, Xbyak::Xmm(zmm_hreduce.getIdx()));
            inc(reg_n);
            cmp(reg_n, reg_k_iter);
            jl(scalar_loop, T_NEAR);
        }
        L(k_tail_done);

        hreduce_vmm_dword(zmm_acc, zmm_hreduce);
        mov(reg_tmp, ptr[reg_param + GET_OFF(T_buf)]);
        vmovd(ptr[reg_tmp + reg_m * i32_sz], Xmm(zmm_acc.getIdx()));

        add(reg_row_ptr, src_m_stride_bytes_);
        inc(reg_m);
        jmp(m_loop, T_NEAR);
        L(m_done);
    }

    // -------------------------------------------------------------------------
    // S-reduce: S_buf[n] = sum_{k ∈ [k_start, k_end)} wei[k, n_base+n].
    //
    // N is the inner (stride-1) axis on wei. Walk N in 16-lane chunks;
    // for each chunk, walk K and accumulate widened bytes with VPADDD
    // into a per-N int32 vector accumulator; masked store on the last
    // partial chunk.
    // -------------------------------------------------------------------------
    void emit_s_reduce() {
        if (wei_is_subbyte_) {
            emit_s_reduce_subbyte();
            return;
        }
        // Chunk base = wei_batch_ptr + k_start*wei_k_stride + n_base*1
        // (dt_sz == 1 for s8/u8 → n offset in bytes equals elements).
        load_param(reg_n_chunk_ptr, GET_OFF(wei_batch_ptr));
        load_param(reg_tmp, GET_OFF(k_start));
        imul(reg_tmp, reg_tmp, wei_k_stride_bytes_);
        add(reg_n_chunk_ptr, reg_tmp);
        load_param(reg_tmp, GET_OFF(n_base));
        add(reg_n_chunk_ptr, reg_tmp);

        xor_(reg_n, reg_n);
        Label n_loop, n_done;
        L(n_loop);
        // remaining = N_blk - n
        mov(reg_tmp, reg_N_blk);
        sub(reg_tmp, reg_n);
        test(reg_tmp, reg_tmp);
        jle(n_done, T_NEAR);

        Label use_full, mask_set;
        cmp(reg_tmp, n_lanes);
        jge(use_full, T_NEAR);
        build_low_bits_mask(reg_tmp);
        jmp(mask_set, T_NEAR);
        L(use_full);
        set_full_mask();
        L(mask_set);

        uni_vpxor(zmm_acc, zmm_acc, zmm_acc);

        mov(reg_row_ptr, reg_n_chunk_ptr);
        mov(reg_k_iter, reg_k_len);
        Label k_loop, k_done;
        L(k_loop);
        test(reg_k_iter, reg_k_iter);
        jz(k_done, T_NEAR);
        if (wei_is_signed_)
            vpmovsxbd(zmm_data, ptr[reg_row_ptr]);
        else
            vpmovzxbd(zmm_data, ptr[reg_row_ptr]);
        vpaddd(zmm_acc, zmm_acc, zmm_data);
        add(reg_row_ptr, wei_k_stride_bytes_);
        dec(reg_k_iter);
        jmp(k_loop, T_NEAR);
        L(k_done);

        // Store n_lanes int32 lanes (masked on partial chunk) to S_buf[n].
        mov(reg_tmp, ptr[reg_param + GET_OFF(S_buf)]);
        masked_store_i32(ptr[reg_tmp + reg_n * i32_sz], zmm_acc);

        add(reg_n, n_lanes);
        add(reg_n_chunk_ptr, n_lanes); // dt_sz=1 → n_lanes bytes
        jmp(n_loop, T_NEAR);
        L(n_done);
    }

    // -------------------------------------------------------------------------
    // Sub-byte (s4/u4) variant of S-reduce.
    //
    // Wei is packed 2 elements per byte along the inner (N) axis, so a
    // 16-element N chunk lives in 8 contiguous bytes. Each K step:
    //   1. Masked QWORD load → 8 dwords (one byte per dword, upper
    //      24 bits zero).
    //   2. Split into low / high nibbles in 2 ZMMs (8 valid lanes each).
    //   3. Sign-extend each lane for s4 via vpslld 28 / vpsrad 28.
    //   4. Interleave with vpermt2d so output[i] = (i%2==0 ? lo[i/2]
    //      : hi[i/2]) — producing 16 dwords in linear N order.
    //   5. vpaddd into the per-N int32 accumulator.
    //
    // The factory enforces N_blk % 2 == 0 and B_strides[1] % 2 == 0 for
    // sub-byte wei so chunk byte/element offsets remain aligned.
    // -------------------------------------------------------------------------
    void emit_s_reduce_subbyte() {
        // Permute table for vpermt2d: dst[i] selects from src1 (= dst,
        // 0–15) or src2 (16–31). We want
        //   {lo[0], hi[0], lo[1], hi[1], …, lo[7], hi[7]}.
        alignas(64) static constexpr const uint32_t int4_interleave_dw[16]
                = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};
        mov(reg_tmp,
                reinterpret_cast<uintptr_t>(
                        static_cast<const void *>(int4_interleave_dw)));
        vmovdqu32(zmm_perm, ptr[reg_tmp]);

        // 0x0000000F per-dword broadcast for nibble mask.
        mov(reg_tmp.cvt32(), 0x0F);
        vpbroadcastd(zmm_0f, reg_tmp.cvt32());

        // Chunk base = wei_batch_ptr + k_start * wei_k_stride_bytes
        //              + n_base / 2 (byte offset for the N origin).
        load_param(reg_n_chunk_ptr, GET_OFF(wei_batch_ptr));
        load_param(reg_tmp, GET_OFF(k_start));
        imul(reg_tmp, reg_tmp, wei_k_stride_bytes_);
        add(reg_n_chunk_ptr, reg_tmp);
        load_param(reg_tmp, GET_OFF(n_base));
        sar(reg_tmp, 1); // 2 elements per byte; n_base is guaranteed even
        add(reg_n_chunk_ptr, reg_tmp);

        constexpr int chunk_bytes = n_lanes / 2; // 16 elems → 8 bytes

        xor_(reg_n, reg_n);
        Label n_loop, n_done;
        L(n_loop);
        // elements_remaining = N_blk - n
        mov(reg_tmp, reg_N_blk);
        sub(reg_tmp, reg_n);
        test(reg_tmp, reg_tmp);
        jle(n_done, T_NEAR);

        // Two masks per chunk:
        //   k_mask      → store mask, set to min(remaining, 16) bits.
        //   k_byte_mask → byte-load mask, set to ceil(min(remaining,16) / 2)
        //                  bits (one bit per packed byte loaded).
        Label use_full, mask_set;
        cmp(reg_tmp, n_lanes);
        jge(use_full, T_NEAR);
        build_low_bits_opmask(k_mask, reg_tmp);

        mov(reg_tmp, reg_N_blk);
        sub(reg_tmp, reg_n);
        // bytes_to_load = (remaining + 1) / 2
        inc(reg_tmp);
        sar(reg_tmp, 1);
        build_low_bits_opmask(k_byte_mask, reg_tmp);
        jmp(mask_set, T_NEAR);
        L(use_full);
        mov(reg_tmp, 0xFFFF);
        kmovw(k_mask, reg_tmp.cvt32());
        mov(reg_tmp, 0xFF);
        kmovw(k_byte_mask, reg_tmp.cvt32());
        L(mask_set);

        uni_vpxor(zmm_acc, zmm_acc, zmm_acc);
        mov(reg_row_ptr, reg_n_chunk_ptr);
        mov(reg_k_iter, reg_k_len);

        Label k_loop, k_done;
        L(k_loop);
        test(reg_k_iter, reg_k_iter);
        jz(k_done, T_NEAR);

        // Load 8 packed bytes → 8 dwords in low half of zmm_data.
        const Ymm ymm_data(zmm_data.getIdx());
        vpmovzxbd(ymm_data | k_byte_mask | T_z, ptr[reg_row_ptr]);

        // Split nibbles. zmm_t1 = low nibbles, zmm_t2 = high nibbles.
        vpandd(zmm_t1, zmm_data, zmm_0f);
        vpsrld(zmm_t2, zmm_data, 4);
        vpandd(zmm_t2, zmm_t2, zmm_0f);

        // s4: sign-extend each lane (bit 3 is the sign of the nibble).
        if (wei_is_signed_) {
            vpslld(zmm_t1, zmm_t1, 28);
            vpsrad(zmm_t1, zmm_t1, 28);
            vpslld(zmm_t2, zmm_t2, 28);
            vpsrad(zmm_t2, zmm_t2, 28);
        }

        // Interleave: zmm_t1 ← {lo[0], hi[0], lo[1], hi[1], …}.
        // vpermt2d uses zmm_t1 as both dest and src1; src2 = zmm_t2.
        vpermt2d(zmm_t1, zmm_perm, zmm_t2);
        vpaddd(zmm_acc, zmm_acc, zmm_t1);

        add(reg_row_ptr, wei_k_stride_bytes_);
        dec(reg_k_iter);
        jmp(k_loop, T_NEAR);
        L(k_done);

        // Store up to 16 int32 lanes (masked on partial chunk).
        mov(reg_tmp, ptr[reg_param + GET_OFF(S_buf)]);
        vmovdqu32(ptr[reg_tmp + reg_n * i32_sz] | k_mask, zmm_acc);

        add(reg_n, n_lanes);
        add(reg_n_chunk_ptr, chunk_bytes);
        jmp(n_loop, T_NEAR);
        L(n_done);
    }

    // -------------------------------------------------------------------------
    // Combine: delta[m,n] += s_src[m] · s_wei[n] ·
    //                        float( z_src[m]·S[n] + z_wei[n]·T[m]
    //                               − G · z_src[m] · z_wei[n] )
    //
    // Outer m loop pre-broadcasts the per-M scalars. Inner n loop walks
    // 16-lane chunks across the AMX-style stripe layout (stripes of LDC
    // floats wide, separated by M_blk·LDC f32 positions).
    // -------------------------------------------------------------------------
    void emit_combine() {
        load_param(reg_delta_base, GET_OFF(delta_ptr));

        Label stripe_loop, stripe_done;
        xor_(reg_iter, reg_iter); // stripe's global N origin
        L(stripe_loop);
        cmp(reg_iter, reg_N_blk);
        jge(stripe_done, T_NEAR);

        // cols = min(LDC, N_blk - iter)
        mov(reg_n, reg_N_blk);
        sub(reg_n, reg_iter);
        Label cols_clamped;
        cmp(reg_n, LDC_);
        jle(cols_clamped, T_NEAR);
        mov(reg_n, LDC_);
        L(cols_clamped);
        // reg_n now = cols in this stripe; we'll use it as the loop bound.

        // For each m in [0, M_blk):
        xor_(reg_m, reg_m);
        Label m_loop, m_done;
        L(m_loop);
        cmp(reg_m, reg_M_blk);
        jge(m_done, T_NEAR);

        // Per-m broadcasts.
        mov(reg_tmp, ptr[reg_param + GET_OFF(s_src_buf)]);
        vbroadcastss(zmm_ss_bc, ptr[reg_tmp + reg_m * f32_sz]);
        if (has_src_zp_) {
            mov(reg_tmp, ptr[reg_param + GET_OFF(z_src_buf)]);
            vpbroadcastd(zmm_zs_bc, ptr[reg_tmp + reg_m * i32_sz]);
        }
        if (has_wei_zp_) {
            mov(reg_tmp, ptr[reg_param + GET_OFF(T_buf)]);
            vpbroadcastd(zmm_T_bc, ptr[reg_tmp + reg_m * i32_sz]);
        }
        if (has_src_zp_ && has_wei_zp_) {
            // G·zs[m] broadcast. vpbroadcastd-from-GPR is AVX-512 only;
            // on AVX2 we hop through an XMM lane (vmovd + vpbroadcastd).
            // G == reg_k_len (k_end - k_start) precomputed at preamble.
            mov(reg_tmp, ptr[reg_param + GET_OFF(z_src_buf)]);
            mov(reg_tmp.cvt32(), ptr[reg_tmp + reg_m * i32_sz]);
            imul(reg_tmp.cvt32(), reg_k_len.cvt32());
            if (isa_has_masks(isa_)) {
                vpbroadcastd(zmm_Gzs_bc, reg_tmp.cvt32());
            } else {
                const Xbyak::Xmm gzs_x(zmm_Gzs_bc.getIdx());
                vmovd(gzs_x, reg_tmp.cvt32());
                vpbroadcastd(zmm_Gzs_bc, gzs_x);
            }
        }

        // delta_row = delta_base + m * LDC * f32
        mov(reg_delta_row, reg_m);
        imul(reg_delta_row, reg_delta_row, LDC_ * f32_sz);
        add(reg_delta_row, reg_delta_base);

        // Inner n loop within this stripe.
        xor_(reg_k_iter, reg_k_iter); // in-stripe column offset
        Label n_loop, n_done;
        L(n_loop);
        cmp(reg_k_iter, reg_n);
        jge(n_done, T_NEAR);

        // global N for this chunk = iter (stripe origin) + k_iter (in-stripe)
        // remaining_in_stripe = n - k_iter
        push(reg_n);
        sub(reg_n, reg_k_iter);
        Label full_mask, mask_set;
        cmp(reg_n, n_lanes);
        jge(full_mask, T_NEAR);
        build_low_bits_mask(reg_n);
        jmp(mask_set, T_NEAR);
        L(full_mask);
        set_full_mask();
        L(mask_set);
        pop(reg_n);

        // Per-N loads. Global N index = iter + k_iter.
        // tmp_n = iter + k_iter
        push(reg_n);
        mov(reg_n, reg_iter);
        add(reg_n, reg_k_iter);
        // Now reg_n is the global N for the chunk base.

        mov(reg_tmp, ptr[reg_param + GET_OFF(s_wei_buf)]);
        masked_load_f32(zmm_sw, ptr[reg_tmp + reg_n * f32_sz]);
        if (has_wei_zp_) {
            mov(reg_tmp, ptr[reg_param + GET_OFF(z_wei_buf)]);
            masked_load_i32(zmm_zw, ptr[reg_tmp + reg_n * i32_sz]);
        }
        if (has_src_zp_) {
            mov(reg_tmp, ptr[reg_param + GET_OFF(S_buf)]);
            masked_load_i32(zmm_S, ptr[reg_tmp + reg_n * i32_sz]);
        }
        pop(reg_n);

        // term = z_src·S + z_wei·T − G·z_src·z_wei.
        if (has_src_zp_ && has_wei_zp_) {
            vpmulld(zmm_t1, zmm_zs_bc, zmm_S);
            vpmulld(zmm_t2, zmm_T_bc, zmm_zw);
            vpmulld(zmm_t3, zmm_Gzs_bc, zmm_zw);
            vpaddd(zmm_term, zmm_t1, zmm_t2);
            vpsubd(zmm_term, zmm_term, zmm_t3);
        } else if (has_src_zp_) {
            vpmulld(zmm_term, zmm_zs_bc, zmm_S);
        } else {
            vpmulld(zmm_term, zmm_T_bc, zmm_zw);
        }
        vcvtdq2ps(zmm_term_f, zmm_term);

        vmulps(zmm_ss_sw, zmm_ss_bc, zmm_sw);
        vmulps(zmm_term_f, zmm_ss_sw, zmm_term_f);

        masked_load_f32(zmm_delta, ptr[reg_delta_row + reg_k_iter * f32_sz]);
        vaddps(zmm_delta, zmm_delta, zmm_term_f);
        masked_store_f32(ptr[reg_delta_row + reg_k_iter * f32_sz], zmm_delta);

        add(reg_k_iter, n_lanes);
        jmp(n_loop, T_NEAR);
        L(n_done);

        inc(reg_m);
        jmp(m_loop, T_NEAR);
        L(m_done);

        // Advance stripe.
        add(reg_iter, LDC_);
        add(reg_delta_base, M_blk_static_ * LDC_ * f32_sz);
        jmp(stripe_loop, T_NEAR);
        L(stripe_done);
    }

    void generate() override {
        preamble();

        // Persistent runtime values.
        load_param(reg_M_blk, GET_OFF(M_blk));
        load_param(reg_N_blk, GET_OFF(N_blk));
        load_param(reg_k_len, GET_OFF(k_end));
        load_param(reg_tmp, GET_OFF(k_start));
        sub(reg_k_len, reg_tmp);

        // AVX2 tail-mask construction reads {0,1,...,n_lanes-1} from a
        // RIP-relative rodata block. Load once into vmm_idx_; reused on
        // every build_low_bits_mask call.
        if (!isa_has_masks(isa_)) { vmovups(vmm_idx_, ptr[rip + idx_table_]); }

        // Optional delta reset on first K-group.
        load_param(reg_tmp, GET_OFF(first_group));
        Label skip_reset;
        test(reg_tmp, reg_tmp);
        jnz(skip_reset, T_NEAR);
        emit_delta_reset();
        L(skip_reset);

        // Per-group work.
        if (has_wei_zp_) emit_t_reduce();
        if (has_src_zp_) {
            Label skip_s_reduce_lbl;
            load_param(reg_tmp, GET_OFF(skip_s_reduce));
            test(reg_tmp, reg_tmp);
            jnz(skip_s_reduce_lbl, T_NEAR);
            emit_s_reduce();
            L(skip_s_reduce_lbl);
        }
        emit_combine();

        postamble();

        align(64);
        L(idx_table_);
        for (int i = 0; i < n_lanes; ++i)
            dd(i);
    }

#undef GET_OFF
};

// -----------------------------------------------------------------------------
// per_mn_comp_kernel_t concrete impl.
// -----------------------------------------------------------------------------

// Reads a scalar zero point honoring per-axis element strides; returns 0
// when `base` is NULL.
inline int load_zp_scalar(const void *base, data_type_t dt, dim_t mn_base,
        dim_t mn_idx, dim_t mn_stride_elems, dim_t g_idx,
        dim_t g_stride_elems) {
    if (base == nullptr) return 0;
    const dim_t elem_idx
            = (mn_base + mn_idx) * mn_stride_elems + g_idx * g_stride_elems;
    return io::load_int_value(dt, base, elem_idx);
}

// Reads a scalar scale; returns 1.0f when `base` is NULL.
inline float load_scale_scalar(const void *base, data_type_t dt, dim_t mn_base,
        dim_t mn_idx, dim_t mn_stride_elems, dim_t g_idx,
        dim_t g_stride_elems) {
    if (base == nullptr) return 1.0f;
    const dim_t elem_idx
            = (mn_base + mn_idx) * mn_stride_elems + g_idx * g_stride_elems;
    return io::load_float_value(dt, base, elem_idx);
}

// Resolved K-group size for the per-mn compensation: the smallest of
// the active per-K group sizes (src/wei scales and zps), or K when no
// axis is per-K. Used to size and walk S_buf groups consistently in
// `scratch_layout_t::from_conf` and the trampoline.
inline dim_t resolve_k_group_size(const brgemm_matmul_conf_t &bgmmc) {
    dim_t k_group_size = bgmmc.K;
    auto pick = [&](bool active, dim_t gsize) {
        if (active && gsize > 0 && gsize < k_group_size) k_group_size = gsize;
    };
    pick(bgmmc.is_wei_scale_per_k, bgmmc.wei_scales_k_gsize);
    pick(bgmmc.is_src_scale_per_k, bgmmc.src_scales_k_gsize);
    pick(bgmmc.is_wei_zp_per_k, bgmmc.wei_zp_k_gsize);
    pick(bgmmc.is_src_zp_per_k, bgmmc.src_zp_k_gsize);
    return k_group_size;
}

inline dim_t num_k_groups_for(const brgemm_matmul_conf_t &bgmmc) {
    return utils::div_up(bgmmc.K, resolve_k_group_size(bgmmc));
}

template <typename Vmm>
struct per_mn_comp_kernel_impl_t : public per_mn_comp_kernel_t {
    per_mn_comp_kernel_impl_t(const brgemm_matmul_conf_t *bgmmc)
        : conf_(bgmmc), gen_(bgmmc) {}

    status_t create_kernel() { return gen_.create_kernel(); }

    void operator()(const ctx_t *ctx) const override;

private:
    const brgemm_matmul_conf_t *conf_;
    jit_per_mn_comp_kernel_t<Vmm> gen_;
};

template <typename Vmm>
void per_mn_comp_kernel_impl_t<Vmm>::operator()(const ctx_t *ctx) const {
    const auto &c = *conf_;
    assert(ctx->delta_ptr != nullptr);
    assert(ctx->M_blk > 0 && ctx->N_blk > 0 && c.K > 0);

    // Unified K-group size = min(active per-K group sizes). Non-per-K
    // axes broadcast.
    const dim_t k_group_size = resolve_k_group_size(c);

    // Per-axis (mn_base, mn_stride_elems, g_stride_elems, k_gsize).
    // Pointer == nullptr ⇒ contribution absent.
    const void *src_zp_ptr = nullptr;
    const data_type_t src_zp_dt = c.src_zp_dt;
    dim_t src_zp_m_base = 0;
    dim_t src_zp_m_stride_elems = 0;
    dim_t src_zp_g_stride_elems = 0;
    dim_t src_zp_k_gsize = 0;
    if (c.has_zero_point_a && ctx->src_zp_ptr != nullptr) {
        src_zp_ptr = ctx->src_zp_ptr;
        if (c.is_src_zp_per_k) {
            const dim_t num_groups_src_zp
                    = utils::div_up(c.K, c.src_zp_k_gsize);
            src_zp_m_base = ctx->m_base;
            src_zp_m_stride_elems = num_groups_src_zp;
            src_zp_g_stride_elems = 1;
            src_zp_k_gsize = c.src_zp_k_gsize;
        }
    }

    const void *wei_zp_ptr = nullptr;
    const data_type_t wei_zp_dt = c.wei_zp_dt;
    dim_t wei_zp_n_base = 0;
    dim_t wei_zp_n_stride_elems = 0;
    dim_t wei_zp_g_stride_elems = 0;
    dim_t wei_zp_k_gsize = 0;
    if (c.has_zero_point_b && ctx->wei_zp_ptr != nullptr) {
        wei_zp_ptr = ctx->wei_zp_ptr;
        if (!c.is_wei_zp_common) {
            wei_zp_n_base = ctx->n_base;
            wei_zp_n_stride_elems = 1;
            wei_zp_g_stride_elems = c.is_wei_zp_per_k ? c.N : 0;
            wei_zp_k_gsize = c.is_wei_zp_per_k ? c.wei_zp_k_gsize : 0;
        }
    }

    // Src scales: per-K folds into delta; common src folds in only when
    // per-K wei scales are active (the brgemm kernel folds it into per-K
    // wei multiply, so the delta must follow).
    const void *src_scales_ptr = nullptr;
    data_type_t src_scales_dt = c.src_scales_dt;
    dim_t src_scales_m_base = 0;
    dim_t src_scales_m_stride_elems = 0;
    dim_t src_scales_g_stride_elems = 0;
    dim_t src_scales_k_gsize = 0;
    if (ctx->src_scales_ptr != nullptr && c.with_src_scales
            && c.is_src_scale_per_k) {
        const dim_t num_groups_src_sc
                = utils::div_up(c.K, c.src_scales_k_gsize);
        src_scales_ptr = ctx->src_scales_ptr;
        src_scales_m_base = ctx->m_base;
        src_scales_m_stride_elems = num_groups_src_sc;
        src_scales_g_stride_elems = 1;
        src_scales_k_gsize = c.src_scales_k_gsize;
    } else if (ctx->src_scales_ptr != nullptr && c.with_src_scales
            && !c.is_src_scale_per_k && c.is_wei_scale_per_k) {
        src_scales_ptr = ctx->src_scales_ptr;
        // m_base = 0, all strides = 0 ⇒ broadcast.
    }

    // Wei scales: per-K folds into delta.
    const void *wei_scales_ptr = nullptr;
    data_type_t wei_scales_dt = c.wei_scales_dt;
    dim_t wei_scales_n_base = 0;
    dim_t wei_scales_n_stride_elems = 0;
    dim_t wei_scales_g_stride_elems = 0;
    dim_t wei_scales_k_gsize = 0;
    if (ctx->wei_scales_ptr != nullptr && c.with_wei_scales
            && !c.apply_scales_in_buffer_b && c.is_wei_scale_per_k) {
        wei_scales_ptr = ctx->wei_scales_ptr;
        wei_scales_n_base = ctx->n_base;
        wei_scales_n_stride_elems = 1;
        wei_scales_g_stride_elems = c.N;
        wei_scales_k_gsize = c.wei_scales_k_gsize;
    }

    const dim_t M_blk = ctx->M_blk;
    const dim_t N_blk = ctx->N_blk;
    const dim_t num_k_groups = utils::div_up(c.K, k_group_size);
    const bool has_src_zp = (src_zp_ptr != nullptr);
    const bool has_wei_zp = (wei_zp_ptr != nullptr);

    auto axis_kg
            = [&](dim_t kg) -> dim_t { return kg > 0 ? kg : k_group_size; };
    const dim_t src_zp_kg = axis_kg(src_zp_k_gsize);
    const dim_t wei_zp_kg = axis_kg(wei_zp_k_gsize);
    const dim_t src_sc_kg = axis_kg(src_scales_k_gsize);
    const dim_t wei_sc_kg = axis_kg(wei_scales_k_gsize);

    // No defensive pre-fills: every K-group below unconditionally
    // overwrites s_*_buf[0..M/N_blk) via load_scale_scalar (returns 1.0
    // when the source ptr is null), and z_*_buf[0..M/N_blk) is written
    // only when has_*_zp is true — which is the only condition under
    // which the JIT reads it.
    typename jit_per_mn_comp_kernel_t<Vmm>::call_params_t p;
    p.src_batch_ptr = ctx->src_batch_ptr;
    p.wei_batch_ptr = ctx->wei_batch_ptr;
    p.T_buf = ctx->T_buf;
    p.S_buf = ctx->S_buf;
    p.z_src_buf = ctx->z_src_buf;
    p.z_wei_buf = ctx->z_wei_buf;
    p.s_src_buf = ctx->s_src_buf;
    p.s_wei_buf = ctx->s_wei_buf;
    p.delta_ptr = ctx->delta_ptr;
    p.m_base = ctx->m_base;
    p.n_base = ctx->n_base;
    p.M_blk = M_blk;
    p.N_blk = N_blk;

    const size_t s_buf_g_stride_lanes = static_cast<size_t>(c.N_blk);
    const dim_t skip_s_reduce = ctx->s_buf_valid ? 1 : 0;

    for (dim_t g = 0; g < num_k_groups; ++g) {
        const dim_t k_start = g * k_group_size;
        const dim_t k_end = nstl::min(k_start + k_group_size, c.K);

        const dim_t g_src_zp = k_start / src_zp_kg;
        const dim_t g_wei_zp = k_start / wei_zp_kg;
        const dim_t g_src_sc = k_start / src_sc_kg;
        const dim_t g_wei_sc = k_start / wei_sc_kg;

        for (dim_t m = 0; m < M_blk; ++m) {
            if (has_src_zp)
                ctx->z_src_buf[m] = load_zp_scalar(src_zp_ptr, src_zp_dt,
                        src_zp_m_base, m, src_zp_m_stride_elems, g_src_zp,
                        src_zp_g_stride_elems);
            ctx->s_src_buf[m] = load_scale_scalar(src_scales_ptr, src_scales_dt,
                    src_scales_m_base, m, src_scales_m_stride_elems, g_src_sc,
                    src_scales_g_stride_elems);
        }
        for (dim_t n = 0; n < N_blk; ++n) {
            if (has_wei_zp)
                ctx->z_wei_buf[n] = load_zp_scalar(wei_zp_ptr, wei_zp_dt,
                        wei_zp_n_base, n, wei_zp_n_stride_elems, g_wei_zp,
                        wei_zp_g_stride_elems);
            ctx->s_wei_buf[n] = load_scale_scalar(wei_scales_ptr, wei_scales_dt,
                    wei_scales_n_base, n, wei_scales_n_stride_elems, g_wei_sc,
                    wei_scales_g_stride_elems);
        }

        p.k_start = k_start;
        p.k_end = k_end;
        p.first_group = (g == 0) ? 0 : 1;
        p.S_buf = ctx->S_buf + g * s_buf_g_stride_lanes;
        p.skip_s_reduce = skip_s_reduce;
        gen_(&p);
    }
}

} // namespace

// -----------------------------------------------------------------------------
// Scratchpad layout
// -----------------------------------------------------------------------------
per_mn_comp_kernel_t::scratch_layout_t
per_mn_comp_kernel_t::scratch_layout_t::from_conf(
        const brgemm_matmul_conf_t &bgmmc) {
    constexpr size_t align = 64;
    scratch_layout_t L;
    const size_t M = static_cast<size_t>(bgmmc.M_blk);
    const size_t N = static_cast<size_t>(bgmmc.N_blk);

    const size_t num_groups = static_cast<size_t>(num_k_groups_for(bgmmc));

    L.T_off_bytes = 0;
    L.S_off_bytes = align_up(L.T_off_bytes + M * sizeof(int32_t), align);
    L.z_src_off_bytes
            = align_up(L.S_off_bytes + N * num_groups * sizeof(int32_t), align);
    L.z_wei_off_bytes
            = align_up(L.z_src_off_bytes + M * sizeof(int32_t), align);
    L.s_src_off_bytes
            = align_up(L.z_wei_off_bytes + N * sizeof(int32_t), align);
    L.s_wei_off_bytes = align_up(L.s_src_off_bytes + M * sizeof(float), align);
    L.total_bytes = align_up(L.s_wei_off_bytes + N * sizeof(float), align);
    return L;
}

void per_mn_comp_kernel_t::scratch_layout_t::pointers_in(
        ctx_t &ctx, char *scratch_base) const {
    ctx.T_buf = reinterpret_cast<int32_t *>(scratch_base + T_off_bytes);
    ctx.S_buf = reinterpret_cast<int32_t *>(scratch_base + S_off_bytes);
    ctx.z_src_buf = reinterpret_cast<int32_t *>(scratch_base + z_src_off_bytes);
    ctx.z_wei_buf = reinterpret_cast<int32_t *>(scratch_base + z_wei_off_bytes);
    ctx.s_src_buf = reinterpret_cast<float *>(scratch_base + s_src_off_bytes);
    ctx.s_wei_buf = reinterpret_cast<float *>(scratch_base + s_wei_off_bytes);
}

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------
status_t per_mn_comp_kernel_t::create(
        std::unique_ptr<per_mn_comp_kernel_t> &kernel,
        const brgemm_matmul_conf_t *bgmmc) {
    // ISA: VPDPBUSD for T-reduce + dword-mask machinery for tails.
    // Supported: AVX-512 + VNNI (Zmm path) or AVX2-VNNI / AVX2-VNNI-2
    // (Ymm path). The dispatched ISA is taken from bgmmc->isa.
    const cpu_isa_t isa = bgmmc->isa;
    const bool isa_ok
            = is_superset(isa, avx512_core_vnni) || is_superset(isa, avx2_vnni);
    if (!isa_ok) return status::unimplemented;

    // Dtype: src is restricted to s8/u8 (existing benchdnn coverage
    // never exercises sub-byte src). Wei accepts s8/u8 plus sub-byte
    // s4/u4 — handled inside emit_s_reduce via the dedicated path.
    if (!utils::one_of(bgmmc->orig_src_dt, data_type::s8, data_type::u8))
        return status::unimplemented;
    if (!utils::one_of(bgmmc->orig_wei_dt, data_type::s8, data_type::u8,
                data_type::s4, data_type::u4))
        return status::unimplemented;
    if (bgmmc->a_dt_sz != 1 || bgmmc->b_dt_sz != 1)
        return status::unimplemented;

    if (bgmmc->A_strides[0] != 1) return status::unimplemented;
    if (bgmmc->B_strides[0] != 1) return status::unimplemented;

    // Sub-byte wei is packed 2 elements per byte along the inner N axis;
    // require even N_blk and an even K-element stride so chunk offsets
    // remain on byte boundaries. The sub-byte S-reduce uses opmask-only
    // primitives (vpmovzxbd with k_byte_mask, vpermt2d), so it is
    // AVX-512-only — reject on AVX2.
    const bool wei_is_subbyte
            = utils::one_of(bgmmc->orig_wei_dt, data_type::s4, data_type::u4);
    if (wei_is_subbyte) {
        if (!isa_has_masks(isa)) return status::unimplemented;
        if (bgmmc->N_blk % 2 != 0) return status::unimplemented;
        if (bgmmc->B_strides[1] % 2 != 0) return status::unimplemented;
    }

    // Combine kernel walks the stripe layout; LDC and M_blk must be
    // valid statics.
    if (bgmmc->LDC <= 0 || bgmmc->M_blk <= 0) return status::unimplemented;

    if (!bgmmc->has_zero_point_a && !bgmmc->has_zero_point_b)
        return status::unimplemented;

    // Dispatch on mask capability: AVX-512 → Zmm; AVX2-VNNI(-2) → Ymm.
    if (isa_has_masks(isa)) {
        auto impl = utils::make_unique<per_mn_comp_kernel_impl_t<Xbyak::Zmm>>(
                bgmmc);
        if (!impl) return status::out_of_memory;
        CHECK(impl->create_kernel());
        kernel = std::move(impl);
    } else {
        auto impl = utils::make_unique<per_mn_comp_kernel_impl_t<Xbyak::Ymm>>(
                bgmmc);
        if (!impl) return status::out_of_memory;
        CHECK(impl->create_kernel());
        kernel = std::move(impl);
    }
    return status::success;
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
