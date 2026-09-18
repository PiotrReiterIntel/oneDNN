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

# Shared setup for every plugin's standalone CMake project in this
# directory -- factored out so a new plugin copied from matmul_template/
# doesn't also copy (and drift from) this boilerplate. A plugin's own
# CMakeLists.txt should, in this order:
#   cmake_minimum_required(VERSION 3.16)
#   project(<name> LANGUAGES CXX)
#   include(${CMAKE_CURRENT_SOURCE_DIR}/../common.cmake)
#   add_library(<target> SHARED <sources>)
#   plugin_common_setup(<target>)

if(NOT ONEDNN_SOURCE_DIR)
    message(FATAL_ERROR
        "ONEDNN_SOURCE_DIR must point at the root of the oneDNN fork this "
        "plugin is built against (the directory containing src/ and "
        "include/).")
endif()
if(NOT ONEDNN_BUILD_DIR)
    message(FATAL_ERROR
        "ONEDNN_BUILD_DIR must point at that oneDNN fork's already-"
        "configured build directory (for generated headers such as "
        "dnnl_config.h/dnnl_version.h) and its built dnnl library.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_CXX_FLAGS MATCHES "-fsycl")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
endif()

# Must match the NDEBUG-ness of the oneDNN build.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_definitions(NDEBUG)

# Call once, after add_library(<target> ...), to apply the include paths,
# dnnl link, and visibility settings every plugin needs identically.
function(plugin_common_setup target)
    target_include_directories(${target} PRIVATE
        ${ONEDNN_SOURCE_DIR}/src
        ${ONEDNN_SOURCE_DIR}/include
        ${ONEDNN_BUILD_DIR}/include
    )

    find_library(ONEDNN_LIB
        NAMES dnnl
        PATHS ${ONEDNN_BUILD_DIR}/src ${ONEDNN_BUILD_DIR}
        NO_DEFAULT_PATH
    )
    if(NOT ONEDNN_LIB)
        message(FATAL_ERROR
            "Could not find the built dnnl library under ${ONEDNN_BUILD_DIR}. "
            "Build oneDNN itself first.")
    endif()
    target_link_libraries(${target} PRIVATE ${ONEDNN_LIB})

    # Plugins link against a host libdnnl.so built with -DDNNL_NATIVE_PLUGIN_HOST=ON.
    # Unresolved native C++ symbols resolve directly from the host library.
    if(UNIX AND NOT APPLE)
        target_link_options(${target} PRIVATE -Wl,--no-undefined)
    endif()

    set_target_properties(${target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endfunction()
