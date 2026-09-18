# oneDNN GPU Plugins

## Intro

This directory provides the oneDNN per-primitive GPU plugin infrastructure.
It allows running custom primitive implementations dynamically loaded from
plugins (shared libraries) at runtime without modifying or recompiling the core
oneDNN library.

When loaded, each plugin registers with oneDNN and its implementations are
injected ahead of oneDNN's own built-in implementations for that primitive
kind, so they get first refusal during dispatch.
See `dnnl_gpu_plugin_register` -- the only symbol a plugin must export.

## Architecture

Plugins are standalone shared libraries (~110 KB) that link dynamically
against a host oneDNN library built with `DNNL_NATIVE_PLUGIN_HOST=ON`.
This lockstep development mode allows plugins to access oneDNN's internal C++
primitive descriptor and memory storage APIs while avoiding duplicated caches
or static runtimes.

```text
[ Application / benchdnn ]
           │
           ▼
     [ libdnnl.so ]  (Host build: DNNL_NATIVE_PLUGIN_HOST=ON)
           │
           │ dlopen(ONEDNN_GPU_PLUGIN_PATH)
           ▼
[ libkf_matmul_plugin.so ] (~110 KB)
  ├── Registers implementation at front of oneDNN matmul list
  ├── Checks applicability in pd_t::init()
  │     ├── Accept: executes evolving SYCL kernel (USM pointers)
  │     └── Reject: returns unimplemented -> oneDNN falls back to native JIT
  └── Shares host primitive cache, engine, and memory storages
```

## Building

### Step 1: Build the Host oneDNN Library

Build the shared oneDNN host with native plugin symbol export enabled:

```bash
source /opt/intel/oneapi/setvars.sh

cmake -S . -B build-native-plugin-host \
      -DDNNL_LIBRARY_TYPE=SHARED \
      -DDNNL_NATIVE_PLUGIN_HOST=ON \
      -DDNNL_GPU_RUNTIME=SYCL \
      -DDNNL_GPU_VENDOR=INTEL
cmake --build build-native-plugin-host --target dnnl tests/benchdnn/benchdnn -j$(nproc)
```

### Step 2: Build the Plugins

Each plugin subdirectory here (e.g. `kf_matmul_plugin`, `matmul_template`) is
its own standalone CMake project sharing the same `common.cmake` setup:

```bash
ONEDNN_BUILD_DIR=$(pwd)/build-native-plugin-host
PLUGIN_DIR=plugins/kf_matmul_plugin   # or plugins/matmul_template, etc.

cmake -B "$PLUGIN_DIR/build" -S "$PLUGIN_DIR" \
      -DONEDNN_SOURCE_DIR=$(pwd) \
      -DONEDNN_BUILD_DIR="$ONEDNN_BUILD_DIR" \
      -DCMAKE_CXX_COMPILER=icpx
cmake --build "$PLUGIN_DIR/build"
```

This produces `$PLUGIN_DIR/build/lib<plugin-target>.so` (e.g.
`libkf_matmul_plugin.so`).

## Sample: Running Both Plugins Together

A minimal end-to-end demo loading both plugins at once to show real dispatch:
the inert template getting first refusal and rejecting, then falling through
to the plugin that actually accepts.

- Build `kf_matmul_plugin` via Step 2 above.
- Build `matmul_template` via the same procedure (its `kernel_supports()`
  always returns `false`, so it is a safe, deterministic "reject everything"
  plugin to demo dispatch order).
- Set `ONEDNN_GPU_PLUGIN_PATH` to both `.so` paths, colon-separated,
  **template first** -- registration order follows `ONEDNN_GPU_PLUGIN_PATH`
  order, so the template is evaluated and rejects before `kf_matmul_plugin`
  gets evaluated.
- Run `benchdnn` against a shape/post-op `kf_matmul_plugin`'s
  `kf_kernel_supports()` accepts (plain 2D f32 `C = relu(A @ B)`), with
  `ONEDNN_VERBOSE=1,dispatch`:

```bash
source /opt/intel/oneapi/setvars.sh

ONEDNN_GPU_PLUGIN_PATH="$(pwd)/plugins/matmul_template/build/libmatmul_template_plugin.so:$(pwd)/plugins/kf_matmul_plugin/build/libkf_matmul_plugin.so" \
ONEDNN_VERBOSE=1,dispatch \
"$ONEDNN_BUILD_DIR/tests/benchdnn/benchdnn" --matmul --engine=gpu \
  --dt=f32 --stag=ab --wtag=ab --dtag=ab \
  --attr-post-ops=relu 32x64:64x128
```

Captured output from this exact command:

```text
onednn_verbose,v1,primitive,create:dispatch,matmul,gpu,matmul,template:matmul:any,undef,
  src:f32::blocked:ab::f0 wei:f32::blocked:ab::f0 dst:f32::blocked:ab::f0,
  attr-post-ops:eltwise_relu,,32x64:64x128,unsupported datatype combination,
  .../plugins/matmul_template/plugin_matmul.cpp:150
  # <-- (A) the template candidate was tried FIRST (ONEDNN_GPU_PLUGIN_PATH
  #     order) and rejected: "create:dispatch" + "template:matmul:any" +
  #     the reject reason ("unsupported datatype combination").

... (CPU-side reorders/fallback attempts by oneDNN's own dispatch, omitted)

onednn_verbose,v1,primitive,exec,gpu,matmul,kf:custom:any,undef,
  src:f32::blocked:ab::f0 wei:f32::blocked:ab::f0 dst:f32::blocked:ab::f0,
  attr-post-ops:eltwise_relu,,32x64:64x128,0
  # <-- (B) the primitive actually CREATED and EXECUTED next:
  #     "exec,gpu,matmul,kf:custom:any" -- the one candidate that accepted,
  #     confirming fallthrough past the template worked and this ran on
  #     "gpu", not a CPU fallback.

0:FAILED (errors:4096 total:4096) (1460 ms) __REPRO: --engine=gpu --matmul
  --impl=kf:custom:any --stag=ab --wtag=ab --dtag=ab --attr-post-ops=relu
  32x64:64x128
  # <-- (C) benchdnn's summary confirms "--impl=kf:custom:any".
  #     The FAILED verdict is an expected accuracy mismatch (got: -nan),
  #     not a dispatch failure -- kf_matmul_plugin ships with its kernel
  #     body still a placeholder for KF intake.
```

## `matmul_template`

A starting point for writing a new matmul plugin. `plugin_matmul_t`/`pd_t`
have the same shape as a real plugin:

- `pd_t::kernel_supports()` returns `false` -- a safe, inert default that
  registers correctly and rejects every shape, so everything falls through to
  oneDNN's own matmul implementations until filled with real criteria.
- `primitive_t::init()`/`execute()` return `status::unimplemented` -- fill
  these with kernel compilation and submission logic (see `kf_matmul.cpp` for a
  worked example).

To start a new plugin from it: copy the directory, rename the
target/namespace/struct names, and rename the
`DECLARE_COMMON_PD_T("template:matmul:any", ...)` string so your kernel is
identifiable in `ONEDNN_VERBOSE` output.

## `kf_matmul_plugin`

The complete, working plugin in this directory demonstrating real dispatch:

- `kf_kernel.sycl.hpp` is regenerated by `update_kernel.sh` from KF's
  `kernel.sycl` (`./update_kernel.sh /path/to/kernel.sycl && cmake --build build`).
  Its evolved body is a host-side function taking a SYCL queue and raw USM
  pointers. `kf_matmul_t::execute()` pulls the queue from oneDNN's stream and
  raw pointers from oneDNN's memory storages.
- `pd_t::kf_kernel_supports()` gates applicability (plain 2D f32 `C = relu(A @ B)`,
  no bias, fixed dims). Any other shape falls through to native oneDNN kernels.

## Registering Multiple Implementations

`dnnl_gpu_plugin_register(plugin_registrar_t &registrar)` can register multiple
implementations for the same primitive kind (e.g. an fp16-specialized and a
bf16-specialized variant) or across different kinds (e.g. matmul and reduction).
The registry keeps a separate list per `primitive_kind_t`.

Multiple plugin `.so`s work identically: list them colon-separated in
`ONEDNN_GPU_PLUGIN_PATH`, and each is loaded and registered in listed order.
Registered implementations precede built-in ones, and the first candidate
whose `pd_t::init()` succeeds is selected.
