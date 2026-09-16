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

# Standalone build for the matmul plugin template. Not wired into the main
# oneDNN CMakeLists.txt -- a plugin is loaded into a running oneDNN process
# via dlopen at runtime (ONEDNN_GPU_PLUGIN_PATH), it is never statically
# linked into libdnnl. See ../README.md for how loading works.
#
# Usage:
#   cmake -B build -DONEDNN_SOURCE_DIR=/path/to/this/oneDNN \
#                  -DONEDNN_BUILD_DIR=/path/to/that/oneDNN/build \
#                  -DCMAKE_CXX_COMPILER=icpx
#   cmake --build build
#
#   ONEDNN_GPU_PLUGIN_PATH=/path/to/build/libmatmul_template_plugin.so \
#     ONEDNN_VERBOSE=1 ./benchdnn --matmul --engine=gpu ...

cmake_minimum_required(VERSION 3.16)
project(matmul_template_plugin LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../common.cmake)

add_library(matmul_template_plugin SHARED
    plugin_matmul.cpp
)

plugin_common_setup(matmul_template_plugin)
