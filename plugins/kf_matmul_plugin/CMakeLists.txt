#===============================================================================
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Standalone build for the example KF matmul GPU plugin. This is
# deliberately NOT wired into the main oneDNN CMakeLists.txt (no
# add_subdirectory(plugins) anywhere in the top-level build) -- a plugin is
# loaded into a running oneDNN process via dlopen at runtime
# (ONEDNN_GPU_PLUGIN_PATH), it is never statically linked into libdnnl.
#
# Because the plugin's headers (kf_matmul.hpp) reach into this oneDNN
# fork's internal src/ headers (gpu_matmul_pd.hpp, sycl_gpu_primitive.hpp,
# ...), it must be built with:
#   - the exact same compiler used for the oneDNN build it will be loaded
#     into (icpx / DPC++, for -fsycl and ABI compatibility -- see
#     gpu_plugin_abi.hpp's same-build-tree caveat),
#   - that oneDNN build's configured source tree AND its build directory
#     (for generated headers such as include/oneapi/dnnl/dnnl_config.h).
#
# Usage:
#   cmake -B build -DONEDNN_SOURCE_DIR=/path/to/this/oneDNN \
#                  -DONEDNN_BUILD_DIR=/path/to/that/oneDNN/build \
#                  -DCMAKE_CXX_COMPILER=icpx
#   cmake --build build
#
#   ONEDNN_GPU_PLUGIN_PATH=/path/to/build/libkf_matmul_plugin.so \
#     ONEDNN_VERBOSE=1 ./benchdnn --matmul --engine=gpu ...

cmake_minimum_required(VERSION 3.16)
project(kf_matmul_plugin LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../common.cmake)

# NDEBUG matching and either legacy archive linking or native host-symbol
# linking are handled uniformly by common.cmake/plugin_common_setup().
add_library(kf_matmul_plugin SHARED
    kf_matmul.cpp
)

plugin_common_setup(kf_matmul_plugin)
