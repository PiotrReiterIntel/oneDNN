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

#ifndef CPU_X64_PLATFORM_HPP
#define CPU_X64_PLATFORM_HPP

#include "cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace platform {

// returns true if the CPU is a hybrid CPU
// (e.g. Meteor Lake, Alder Lake, Raptor Lake, Lunar Lake)
bool is_hybrid();

// Returns true if this hybrid CPU has a low-power E-core island (LP E-cores),
// i.e. Efficient cores that have no L3 cache. Only meaningful on hybrid CPUs.
bool has_lpe_core();

// Returns the per-core cache size in bytes for the given level (1=L1d, 2=L2, 3=L3).
// On non-hybrid systems uses CPUID leaf 4 directly.
// On hybrid systems uses CpuTopology and returns the minimum per-core size across
// all core types (conservative: avoids overflowing the smallest cache present).
unsigned get_per_core_cache_size(int level);

} // namespace platform
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
