# oneDNN Downstream Integration & GPU Plugin Architecture

## Overview

This document tracks major downstream users of oneDNN on GitHub, their integration/linking mechanisms, and the architectural requirements for loading oneDNN GPU SYCL plugins when oneDNN is statically linked into downstream libraries or binaries.

---

## Downstream oneDNN Consumers on GitHub

| Project | Repository | Component / Role | Fetch Mechanism | Linking Mode | Target Backends / API |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **benchdnn** | `oneapi-src/oneDNN` | Reference test & benchmark harness | In-tree (`tests/benchdnn`) | Dynamic (`libdnnl.so`, default) | All oneDNN backends (CPU, Intel GPU SYCL/OCL) |
| **PyTorch** | `pytorch/pytorch` | ATen CPU/XPU & Inductor | Git Submodule (`third_party/ideep`) | Static (into `libtorch_cpu`/`xpu`) | CPU (x86, AArch64/ACL), Intel GPU (SYCL) |
| **OpenVINO** | `openvinotoolkit/openvino` | Intel CPU & GPU plugins | In-tree submodule (`thirdparty/onednn`) | Static (renamed symbols) | CPU (x86, ARM, RISC-V), GPU (OpenCL/SYCL) |
| **ONNX Runtime** | `microsoft/onnxruntime` | DNNL Execution Provider | CMake `ExternalProject_Add` | Dynamic (`libdnnl.so` / `dnnl.dll`) | CPU, GPU (OpenCL), ACL |
| **TensorFlow / XLA** | `tensorflow/tensorflow` | TF MKL ops & XLA CPU | Bazel `http_archive` (`@xla//third_party/mkl_dnn`) | Static (compiled in-tree) | CPU (x86, AArch64) |
| **Intel Ext. for PyTorch (IPEX)** | `intel/intel-extension-for-pytorch` | CPU/XPU core engine | Git Submodule | Static / Dynamic | CPU, GPU (SYCL), oneDNN Graph API |
| **Intel Ext. for TF (ITEX)** | `intel/intel-extension-for-tensorflow` | CPU/GPU ops & Graph API | Bazel `http_archive` | Static | CPU, GPU (SYCL), oneDNN Graph API |
| **GGML / llama.cpp** | `ggml-org/ggml` | `ggml-sycl` Intel GPU backend | CMake `find_package(DNNL)` | Dynamic (`DNNL::dnnl`) | Intel GPU (SYCL) |
| **PaddlePaddle** | `PaddlePaddle/Paddle` | Phi MKL-DNN kernel library | CMake `ExternalProject` / `find_package` | Static / Dynamic | CPU (x86, ARM) |
| **Apache TVM** | `apache/tvm` | BYOC DNNL partitioner & runtime | CMake `find_package(DNNL)` | Dynamic | CPU |
| **OpenCV** | `opencv/opencv` | DNN module SYCL target | CMake `find_package(dnnl)` | Dynamic | CPU / GPU (SYCL) |
| **ktransformers** | `kvcache-ai/ktransformers` | MoE & VNNI CPU inference | CMake `find_package` / Subdirectory | Static / Dynamic | CPU (x86 VNNI/AMX) |

---

## Static Linking vs. oneDNN GPU Plugin Architecture

### Background

oneDNN GPU plugins (e.g. `libkf_matmul_plugin.so`) are standalone shared libraries loaded at runtime via `dlopen` based on `ONEDNN_GPU_PLUGIN_PATH`. Plugins link against the host library to resolve internal native C++ symbols (`dnnl::impl::*`), including:
- Primitive descriptors (`pd_t`)
- Memory storages and stream contexts
- Internal dispatch tables and registrations (`plugin_registrar_t`)

This architecture avoids duplicating oneDNN internal runtime caches and static state across the host and the plugin.

### Implications for Statically Linked Consumers

When downstream projects (such as PyTorch, OpenVINO, or TensorFlow) statically compile and link oneDNN into their own shared objects or executables (`.so` / `.dll`), the internal oneDNN symbols are by default hidden or stripped. To load and resolve oneDNN GPU plugins in these environments, build scripts and link configurations must be updated:

1. **Expose Internal oneDNN Symbols (`-fvisibility=default`)**:
   - oneDNN must be built with `DNNL_NATIVE_PLUGIN_HOST=ON`.
   - By default, non-plugin oneDNN builds compile with `-fvisibility=internal` (or hidden). Downstream build scripts must ensure internal symbols (`dnnl::impl::*`) are not hidden.
   - Any linker version scripts or export symbol lists (e.g., `--version-script`, export maps) must be updated to export `dnnl::impl::*` and `dnnl_gpu_plugin_*` symbols.

2. **Enable Dynamic Symbol Lookup (`-rdynamic` / `RTLD_GLOBAL`)**:
   - When a plugin is dynamically loaded with `dlopen()`, its unresolved references to host oneDNN symbols must be resolvable from the global symbol table.
   - Downstream shared libraries must be loaded with `RTLD_GLOBAL`, or main executables linked with `-rdynamic` (or `-Wl,--export-dynamic`).

3. **Prevent Dead Code Elimination (`--gc-sections`)**:
   - Compilers and linkers frequently discard unused internal oneDNN symbols during static link steps if the downstream library does not directly call them.
   - Linker options may need `--no-gc-sections` or explicit symbol preservation flags (`--undefined` / `-u`) for oneDNN plugin interface symbols.

4. **Address Symbol Renaming / Mangling**:
   - Certain projects (e.g., OpenVINO's static CPU plugin build) rename oneDNN symbols using preprocessor definitions to avoid collisions.
   - Plugins compiled against standard oneDNN headers will fail to resolve renamed symbols at load time unless the plugin is compiled against the exact same renamed definitions.

### Recommended Approaches

- **Option A (Shared Library Link)**: Transition downstream targets or configurations to dynamically link against a shared `libdnnl.so` built with `DNNL_NATIVE_PLUGIN_HOST=ON`. This cleanly exposes all necessary symbols without modifying downstream export lists.
- **Option B (Expose Static Symbols)**: For consumers requiring static integration, add build-time options to downstream build systems to enable `DNNL_NATIVE_PLUGIN_HOST=ON`, apply `-rdynamic`, and export `dnnl::impl::*` symbols from the resulting binary.
