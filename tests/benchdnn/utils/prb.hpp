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

#ifndef UTILS_PRB_HPP
#define UTILS_PRB_HPP

#include "dnn_types.hpp"
#include "utils/impl_filter.hpp"
#include "utils/res.hpp"
#include "utils/wrapper.hpp"

#include <cassert>
#include <string>

// A base class for all driver-specific `prb_t` problem descriptors. It holds
// the members that are common across every driver and exposes a polymorphic
// interface so that functions in `dnnl_common.hpp` can operate on the base type
// without being templated on the concrete driver `prb_t`.
struct base_prb_t {
    base_prb_t() = default;
    base_prb_t(dir_t dir, bool inplace, const attr_t &attr,
            const impl_filter_t &impl_filter)
        : dir(dir), inplace(inplace), attr(attr), impl_filter(impl_filter) {}
    virtual ~base_prb_t() = default;

    dir_t dir = FLAG_FWD;
    bool inplace = false;
    attr_t attr;
    impl_filter_t impl_filter;

    // Used to construct memory desc when dimensions are runtime since such mds
    // can't be used directly from query and memory objects can't be
    // constructed. Drivers supporting runtime dimensions override it.
    virtual benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> get_md(int arg) const {
        assert(!"No runtime dimensions support for this driver!");
        return make_benchdnn_dnnl_wrapper<dnnl_memory_desc_t>(nullptr);
    }

    const char *str() const { return repro.c_str(); }

protected:
    std::string repro;
};

#endif
