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

#include <algorithm>
#include "common/utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/platform.hpp"
#include "xbyak/xbyak_util.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace platform {

namespace {

#ifndef __APPLE__
// Global CPU topology instance, initialized on first use
// This is thread-safe due to C++11 magic statics
// NOTE: xbyak::util::CpuTopology is not supported on macOS.
struct cpu_topology_cach_t {
    Xbyak::util::CpuTopology topology;
    cpu_topology_cach_t() : topology(cpu()) {}
};

// Thread-safe lazy initialization using C++11 magic statics
cpu_topology_cach_t &get_topology_cache() {
    static cpu_topology_cach_t cache;
    return cache;
}

// Map platform cache level (1=L1d, 2=L2, 3=L3) to xbyak CacheType
Xbyak::util::CacheType convert_cache_level(int level) {
    switch (level) {
        case 1: return Xbyak::util::L1d;
        case 2: return Xbyak::util::L2;
        case 3: return Xbyak::util::L3;
        default: return Xbyak::util::CACHE_UNKNOWN;
    }
}

// Filter used when searching for Efficient cores to distinguish between
// regular E-cores (those sharing an L3) and LP E-cores (no L3 present).
// For non-Efficient core types this filter is ignored.
enum class l3_filter_t {
    any, // no L3 filtering
    with_l3, // only cores that have a non-zero L3 (regular E-cores)
    without_l3, // only cores with L3 size == 0 (LP E-cores)
};

// Find the index of a representative logical CPU of the given core type.
// For Efficient cores, l3_filter optionally restricts matches by L3 presence.
// Returns the first matching CPU index, or SIZE_MAX if none is found.
size_t find_representative_cpu(Xbyak::util::CoreType target_type,
        l3_filter_t l3_filter = l3_filter_t::any) {
    const auto &topo = get_topology_cache().topology;
    size_t num_cpus = topo.getLogicalCpuNum();

    for (size_t i = 0; i < num_cpus; i++) {
        const auto &logi = topo.getLogicalCpu(i);
        if (logi.coreType != target_type) continue;

        // Apply L3 filter for Efficient cores only
        if (target_type == Xbyak::util::Efficient
                && l3_filter != l3_filter_t::any) {
            const auto &l3 = topo.getCache(i, Xbyak::util::L3);
            bool has_l3 = (l3.size > 0);
            if (l3_filter == l3_filter_t::with_l3 && !has_l3) continue;
            if (l3_filter == l3_filter_t::without_l3 && has_l3) continue;
        }
        return i;
    }
    return SIZE_MAX; // not found
}

// Calculate per-core cache size for a specific cache level and CPU index.
// Matches the legacy getCoresSharingDataCache semantics: divides by the number
// of PHYSICAL cores sharing the cache, not logical CPUs.  The legacy Xbyak Cpu
// path (CPUID leaf 4) uses L1d sharing count as the SMT width and divides it
// out of every level's logical-CPU sharing count.  We replicate that here so
// that non-hybrid results are identical to the legacy path.
uint32_t calculate_per_core_cache(size_t cpu_index, int level) {
    const auto &topo = get_topology_cache().topology;

    Xbyak::util::CacheType cache_type = convert_cache_level(level);
    if (cache_type == Xbyak::util::CACHE_UNKNOWN) { return 0; }

    const auto &cache = topo.getCache(cpu_index, cache_type);
    if (cache.size == 0) { return 0; }

    // Number of logical CPUs (threads) sharing this cache instance.
    size_t sharing_logical = cache.getSharedCpuNum();
    if (sharing_logical == 0) sharing_logical = 1;

    // SMT width = logical CPUs sharing L1d (L1 is always private to one
    // physical core, so this count equals the number of HT threads per core).
    size_t smt_width
            = topo.getCache(cpu_index, Xbyak::util::L1d).getSharedCpuNum();
    if (smt_width == 0) smt_width = 1;

    // Physical cores sharing this cache (mirrors legacy smt_width division).
    size_t sharing_cores = std::max(sharing_logical / smt_width, size_t(1));

    return cache.size / sharing_cores;
}

#endif // !__APPLE__

} // anonymous namespace

bool is_hybrid() {
#ifdef __APPLE__
    // xbyak::util::CpuTopology is not supported on macOS; assume non-hybrid.
    return false;
#else
    // Use the HYBRID bit from the already-cached Xbyak::Cpu instance
    // (CPUID leaf 7 EDX[15]) to avoid triggering CpuTopology init.
    return cpu().has(Xbyak::util::Cpu::tHYBRID);
#endif
}

// CPUID-based implementation using Xbyak::util::Cpu (leaf 4).
// Used on non-hybrid systems (avoiding CpuTopology init cost) and as the
// fallback for cache_sizing_policy_t::legacy on hybrid systems.
unsigned get_per_core_cache_size_cpuid(int level) {
    if (level > 0 && (unsigned)level <= cpu().getDataCacheLevels()) {
        unsigned l = level - 1;
        return cpu().getDataCacheSize(l) / cpu().getCoresSharingDataCache(l);
    }
    return 0;
}

// Inner implementation: resolves sizing_policy to a cache size with no env-var override.
// Called by both get_per_core_cache_size (which may apply the override first)
// and by topology-info helpers (get_per_core_cache_size_pcore etc.) that must
// always return true topology values regardless of the active env-var override.
static unsigned get_per_core_cache_size_for_policy(
        int level, cache_sizing_policy_t sizing_policy) {
    // Validate level
    if (level < 1 || level > 3) { return 0; }

#ifdef __APPLE__
    return get_per_core_cache_size_cpuid(level);
#else
    if (sizing_policy == cache_sizing_policy_t::legacy) {
        return get_per_core_cache_size_cpuid(level);
    }

    // Fast path: on non-hybrid systems CpuTopology and legacy CPUID return the
    // same value.  Avoid the expensive CpuTopology init on non-hybrid systems.
    if (!is_hybrid()) { return get_per_core_cache_size_cpuid(level); }

    size_t pcore_cpu = find_representative_cpu(Xbyak::util::Performance);
    size_t lp_core_cpu = find_representative_cpu(
            Xbyak::util::Efficient, l3_filter_t::with_l3);
    size_t lpe_core_cpu = find_representative_cpu(
            Xbyak::util::Efficient, l3_filter_t::without_l3);

    if (pcore_cpu == SIZE_MAX) pcore_cpu = 0;
    if (lp_core_cpu == SIZE_MAX) lp_core_cpu = 0;

    uint32_t pcore_size = calculate_per_core_cache(pcore_cpu, level);
    uint32_t lp_core_size = calculate_per_core_cache(lp_core_cpu, level);
    uint32_t lpe_core_size = (lpe_core_cpu != SIZE_MAX)
            ? calculate_per_core_cache(lpe_core_cpu, level)
            : 0;

    switch (sizing_policy) {
        case cache_sizing_policy_t::p_core: return pcore_size;
        case cache_sizing_policy_t::lp_core: return lp_core_size;
        case cache_sizing_policy_t::lpe_core: return lpe_core_size;
        case cache_sizing_policy_t::min: {
            uint32_t m = (std::min)(pcore_size, lp_core_size);
            if (lpe_core_cpu != SIZE_MAX && lpe_core_size > 0)
                m = (std::min)(m, lpe_core_size);
            return m;
        }
        default: return get_per_core_cache_size_cpuid(level);
    }
#endif
}

unsigned get_per_core_cache_size(
        int level, cache_sizing_policy_t sizing_policy) {
    // Check for env-var override (ONEDNN_CACHE_POLICY / DNNL_CACHE_POLICY).
    // Parsed once at first call; ONEDNN_ takes precedence per library convention.
    static const auto policy_override
            = []() -> std::pair<bool, cache_sizing_policy_t> {
        const std::string val = getenv_string_user("CACHE_POLICY");
        if (val == "min") return {true, cache_sizing_policy_t::min};
        if (val == "p_core") return {true, cache_sizing_policy_t::p_core};
        if (val == "lp_core") return {true, cache_sizing_policy_t::lp_core};
        if (val == "lpe_core") return {true, cache_sizing_policy_t::lpe_core};
        if (val == "legacy") return {true, cache_sizing_policy_t::legacy};
        return {false, cache_sizing_policy_t::min};
    }();
    if (policy_override.first) sizing_policy = policy_override.second;

    return get_per_core_cache_size_for_policy(level, sizing_policy);
}

unsigned get_per_core_cache_size_topology(
        int level, cache_sizing_policy_t sizing_policy) {
    return get_per_core_cache_size_for_policy(level, sizing_policy);
}

bool has_lpe_core() {
#ifdef __APPLE__
    return false;
#else
    if (!is_hybrid()) return false;
    return find_representative_cpu(
                   Xbyak::util::Efficient, l3_filter_t::without_l3)
            != SIZE_MAX;
#endif
}

} // namespace platform
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
