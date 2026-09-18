# plugins/ -- agent guide

This file is written for an AI coding agent working in this directory, not
as end-user prose. If you're an agent about to touch anything under
`plugins/`, read this first. There's a short section at the bottom to
adapt when a human asks what this directory does or how to use it.

## What this is

GPU plugins for oneDNN: shared libraries that add GPU primitive
implementations, loaded into a running oneDNN process at load time.
`plugins/` is **not** part of the main oneDNN build -- there is no
`add_subdirectory(plugins)` anywhere in the top-level `CMakeLists.txt`.
Each plugin here is its own standalone CMake project, built separately
against this source tree (same compiler, same headers -- see "ABI" below).

Registered implementations sit **before** oneDNN's own built-in ones in
that primitive kind's implementation list -- first `pd_t::init()` to
return `status::success` wins, exactly like any other oneDNN impl list.
Rejecting (returning anything else) falls through to the next plugin, then
to oneDNN's own implementations. This is real dispatch, not a hijacked
reference-primitive execute path -- a plugin gets the same scratchpad/
format-negotiation/attr access any built-in implementation gets.

## How plugins load

- `ONEDNN_GPU_PLUGIN_PATH` = one or more `.so` paths, separated by `:`.
- The first GPU primitive-implementation-list request in the process loads
  every path in that list, left to right, via `dlopen` + `dlsym`, looking
  for the symbol `dnnl_gpu_plugin_register` (`src/gpu/gpu_plugin_abi.hpp`).
- Each plugin's `dnnl_gpu_plugin_register(plugin_registrar_t&)` calls
  `registrar.register_impl(primitive_kind, item)` once per implementation
  it wants to add -- a single `.so` can register more than one, for the
  same or different primitive kinds.
- Plugins are never `dlclose`d once loaded. There is no hot-reload: a
  changed plugin means rebuild the `.so` and restart the process loading
  it (e.g. `benchdnn`).
- **No cross-build ABI stability.** A plugin must be built with the same
  compiler and against the same oneDNN source/build tree it will be
  loaded into. This is a native C++ interface, not a stable plugin ABI --
  don't add versioning/compat shims for this; rebuild instead.

## What's here

- `common.cmake` -- shared CMake setup (`ONEDNN_SOURCE_DIR`/
  `ONEDNN_BUILD_DIR` validation, standard/PIC/`-fsycl` flags, `NDEBUG`/
  `Release` build-type defaulting, and a `plugin_common_setup(<target>)`
  function for include dirs, the dnnl dynamic link against `libdnnl.so`,
  and visibility settings) that every plugin's own `CMakeLists.txt`
  `include()`s. Keeps that boilerplate from drifting across plugins as more
  get added -- see either existing `CMakeLists.txt` for the include-then-call
  pattern.
- `kf_matmul_plugin/` -- a complete, working matmul plugin. It dispatches
  straight to a real evolvable SYCL kernel body
  (`kf_kernel.sycl.hpp`, digested from an external kernel-generation
  pipeline's own kernel file) rather than a stand-in. Read this one to see
  every part of the mechanism exercised end to end: registration,
  accept/reject hardcoded to one real task's shape/dtype/post-op
  constraints, and execution via raw USM pointers pulled straight from
  oneDNN's own memory storages and stream. Its own file comments explain
  the USM-vs-buffer-memory constraint that comes with reusing a
  raw-pointer kernel signature as-is.
- `matmul_template/` -- a starting point for a *new* matmul plugin. Same
  file layout as `kf_matmul_plugin`, kernel-specific parts left as
  `TODO`s in the `.hpp`/`.cpp` comments themselves (no separate README --
  see the checklist below).

Both are standalone CMake projects -- see each directory's
`CMakeLists.txt`.

## Adding a new plugin (agent checklist)

Only do this when actually asked to add a plugin, not speculatively.

1. Copy `matmul_template/` to a new directory, rename the target and
   `namespace`/struct names (`plugin_matmul_t` etc.) to something specific
   to the kernel being added.
2. In the new `pd_t`:
   - Rename the `DECLARE_COMMON_PD_T("template:matmul:any", ...)` string
     to something that identifies this kernel in `ONEDNN_VERBOSE` output.
   - Fill in `kernel_supports()` with this kernel's real accept/reject
     rule, checked directly against `src_md()`/`weights_md()`/`dst_md()`/
     `attr()` -- no separate plain-data contract to keep in sync. Leaving
     it returning `false` is a safe, inert starting state (registers
     correctly, rejects every shape, everything falls through to oneDNN's
     own impls) -- don't invent a placeholder accept rule just to have
     something there.
   - Book scratchpad via `scratchpad_registry()` in `init()` if the kernel
     needs workspace (see `src/gpu/intel/matmul/gemm.cpp` for the pattern;
     `ref_matmul_t` itself doesn't need one).
3. In `primitive_t`: fill in `init()` (compile/register the kernel,
   typically `create_kernel()`) and `execute()` (submit it). Two real
   submission shapes exist in this tree already -- copy whichever matches
   how the kernel is packaged:
   - A SYCL kernel *class* submitted through oneDNN's own stream/queue via
     `create_kernel()` + `parallel_for(ctx, kernel_, ...)` -- see
     `src/gpu/generic/sycl/ref_matmul.cpp` for the canonical version of
     this shape (what `kf_matmul_plugin` used before it switched to raw
     dispatch).
   - A plain function taking raw USM pointers that submits its own work --
     see `kf_matmul_plugin/kf_matmul.cpp`'s `execute()`. Only works when
     oneDNN's GPU memory is USM-backed, not buffer-backed (benchdnn
     defaults to USM; don't assume the same is true in every caller).
4. Do **not** add a portable/oneDNN-independent capability struct as a
   translation layer for `kernel_supports()` unless specifically asked.
   It was tried and backed out here: it's easy to under-model a kernel's
   real limits (e.g. collapsing "which post-ops does this kernel fuse"
   into one bool) and it duplicates state that's already sitting right
   there on `pd_t`.
5. Add a `CMakeLists.txt` mirroring `matmul_template/CMakeLists.txt`
   (`project()`, `include(../common.cmake)`, `add_library(...)`,
   `plugin_common_setup(<target>)`) -- do not re-duplicate the
   `ONEDNN_SOURCE_DIR`/`ONEDNN_BUILD_DIR` checks, include dirs, dnnl link,
   or `NDEBUG`/`Release` build-type setting that
   `common.cmake`/`plugin_common_setup()` already provide to every plugin
   uniformly -- this is not `kf_matmul_plugin`-specific extra setup to opt
   into, every plugin needs it (see `common.cmake`'s comments for why).
6. Verify both directions with benchdnn, not just the accept path:
   ```
   ONEDNN_GPU_PLUGIN_PATH=/path/to/libyourplugin.so ONEDNN_VERBOSE=1 \
     ./benchdnn --matmul --engine=gpu <args a supported shape> <args>
   ```
   Confirm the verbose `create:gpu` line's `impl=` field shows the new
   plugin's `DECLARE_COMMON_PD_T` name, not `sycl:ref:any` or a built-in
   name. Then rerun with a shape/dtype/attr combo the kernel rejects and
   confirm verbose shows fallthrough to a different impl -- a plugin that
   only ever accepts has an untested reject path.

## Explaining this to a user

Adapt this when a human (not another agent) asks what a plugin here does
or how to run one:

> This is a oneDNN GPU plugin: a small shared library (`.so`) that adds a
> custom kernel implementation for one primitive (e.g. matmul), loaded
> into oneDNN at runtime instead of being compiled into oneDNN itself. To
> use one, build its `.so` (each plugin directory has its own
> `CMakeLists.txt`) and point `ONEDNN_GPU_PLUGIN_PATH` at it before
> running anything that uses oneDNN's GPU path -- `benchdnn`, an
> application linked against this oneDNN build, etc. oneDNN tries it
> first for matching shapes; anything it doesn't handle falls through to
> oneDNN's normal implementations automatically, so it's safe to leave
> loaded even for workloads outside what the plugin covers. Every time
> the plugin's source changes, its `.so` has to be rebuilt and the process
> restarted -- there's no hot-reload, and no ABI stability across
> different oneDNN builds or compilers, so the plugin and the oneDNN build
> it's loaded into must come from the same source/build tree and compiler.
