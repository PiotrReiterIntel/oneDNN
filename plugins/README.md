# oneDNN Plugins

## Intro

This is a POC of oneDNN per-primitive plugin infrastructure, currently
GPU-only.
It allows running custom primitive implementations dynamically loaded from
plugins (shared libraries).
When loaded, each plugin registers with oneDNN and its implementations are
injected ahead of oneDNN's own built-in implementations for that primitive
kind, so they get first refusal during dispatch.
See `dnnl_gpu_plugin_register` -- the only symbol a plugin must export.


## Building

This procedure is the same for every plugin subdirectory here (e.g.
`kf_matmul_plugin`, `matmul_template`) -- each is its own standalone CMake
project sharing the same `common.cmake` setup. Run these from the root of
oneDNN (the directory containing `src/` and `include/`, and this `plugins/`
directory), with `ONEDNN_BUILD_DIR` pointing at an oneDNN build directory
that has already been **built**, not just configured:

```bash
ONEDNN_BUILD_DIR=/home/preiter/onednn-build-fork
PLUGIN_DIR=plugins/kf_matmul_plugin   # or plugins/matmul_template, etc.

source /opt/intel/oneapi/setvars.sh
cmake -B "$PLUGIN_DIR/build" -S "$PLUGIN_DIR" \
      -DONEDNN_SOURCE_DIR=$(pwd) \
      -DONEDNN_BUILD_DIR="$ONEDNN_BUILD_DIR" \
      -DCMAKE_CXX_COMPILER=icpx
cmake --build "$PLUGIN_DIR/build"
```

This produces `$PLUGIN_DIR/build/lib<plugin-target>.so` (e.g.
`libkf_matmul_plugin.so`).

Every plugin here statically links internal oneDNN object code (see
`kf_matmul_plugin`'s section below for why -- in short, `-fvisibility=internal`
means plain dynamic linking against `libdnnl.so` isn't enough, even for
fully generic pd code that never touches a raw kernel pointer). This needs
one extra step **before** the `cmake --build` above: package the main
oneDNN build's already-compiled object code into a static archive, via the
shared `plugins/make_archives.sh`. It reuses `ONEDNN_BUILD_DIR`'s already-
compiled `.o` files (a configure-only build directory won't work) and
writes the archive under `ONEDNN_BUILD_DIR/archives`, so it only needs
rerunning when that build directory itself has been rebuilt, not every time
you rebuild a plugin:

```bash
bash plugins/make_archives.sh "$ONEDNN_BUILD_DIR"
```

## Sample: running both plugins together

A minimal end-to-end demo, straight after Building above -- loads both
plugins at once and shows real dispatch (not two isolated demos): the inert
template getting first refusal and rejecting, then falling through to the
one plugin that actually accepts.

**First, actually (re)build oneDNN itself** and make sure that's the build
both plugins get built against and the one benchdnn below actually loads --
building a plugin against a stale `ONEDNN_BUILD_DIR` (or running benchdnn
with an `LD_LIBRARY_PATH` pointing at a different `libdnnl.so`) silently
produces a plugin that either fails to load or doesn't reflect your current
source:

```bash
cd /path/to/onednn-build-fork   # an already-configured CMake build dir
ninja -j$(nproc) dnnl tests/benchdnn/benchdnn
git -C /path/to/this/oneDNN/checkout rev-parse HEAD   # cross-check against
                                                       # what the build dir
                                                       # was configured from
```

`ninja -n dnnl` immediately after should report no pending `Building CXX
object` steps (on `/mnt/c` under WSL2, a pending relink/symlink step alone
is a benign mtime-granularity artifact of that filesystem, not real drift --
what matters is the absence of compile steps).

Then:

- Build `kf_matmul_plugin` via the Building procedure above (including its
  `make_archives.sh` step).
- Build `matmul_template` via the same procedure, unmodified -- its
  `kernel_supports()` always returns `false`, so it's a safe, deterministic
  "reject everything" plugin to demo ordering against.
- Set `ONEDNN_GPU_PLUGIN_PATH` to both `.so` paths, colon-separated,
  **template first** -- registration order is `ONEDNN_GPU_PLUGIN_PATH`
  order, so this guarantees the template is tried, and seen to reject, before
  `kf_matmul_plugin` ever gets a turn.
- Run benchdnn against a shape/post-op `kf_matmul_plugin`'s
  `kf_kernel_supports()` accepts (plain 2D f32 `C = relu(A @ B)`), with
  `ONEDNN_VERBOSE=1,dispatch` -- the `dispatch` flag is what surfaces each
  candidate's own accept/reject line, not just the final winner -- and
  `ONEDNN_PRIMITIVE_CACHE_CAPACITY=0` (required due to the plugins'
  statically-linked pd cache -- see the `kf_matmul_plugin` section below):

```bash
source /opt/intel/oneapi/setvars.sh   # if not still sourced from Building

ONEDNN_GPU_PLUGIN_PATH="$(pwd)/plugins/matmul_template/build/libmatmul_template_plugin.so:$(pwd)/plugins/kf_matmul_plugin/build/libkf_matmul_plugin.so" \
ONEDNN_VERBOSE=1,dispatch \
ONEDNN_PRIMITIVE_CACHE_CAPACITY=0 \
"$ONEDNN_BUILD_DIR/tests/benchdnn/benchdnn" --matmul --engine=gpu \
  --dt=f32 --stag=ab --wtag=ab --dtag=ab \
  --attr-post-ops=relu 32x64:64x128
```

Real captured output from this exact command (line-wrapped here for
readability; each is one CSV line on stdout), with markers on the three
lines that prove the claims above:

```
onednn_verbose,v1,primitive,create:dispatch,matmul,gpu,matmul,template:matmul:any,undef,
  src:f32::blocked:ab::f0 wei:f32::blocked:ab::f0 dst:f32::blocked:ab::f0,
  attr-post-ops:eltwise_relu,,32x64:64x128,unsupported datatype combination,
  .../plugins/matmul_template/plugin_matmul.cpp:150
  # <-- (A) the template candidate was tried FIRST (ONEDNN_GPU_PLUGIN_PATH
  #     order) and rejected: "create:dispatch" + "template:matmul:any" +
  #     the reject reason ("unsupported datatype combination", the
  #     VDISPATCH_MATMUL message plugin_matmul.cpp's always-false
  #     kernel_supports() falls through to) + the source line inside
  #     matmul_template itself, not any built-in oneDNN impl.

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
  # <-- (C) benchdnn's own summary line, independently confirming (B):
  #     "--impl=kf:custom:any" is the __REPRO benchdnn generates from
  #     whichever impl it actually exercised. The FAILED verdict itself is
  #     an expected correctness mismatch (got: -nan), not a dispatch
  #     failure -- kf_matmul_plugin ships with its kernel body still a
  #     placeholder; that's a KF-side kernel concern, unrelated to the
  #     dispatch mechanism this sample demonstrates.
```

Note the template's rejection reason string is `unsupported datatype
combination` (the `VERBOSE_UNSUPPORTED_DT_CFG` message `VDISPATCH_MATMUL`
attaches to `kernel_supports()`'s `false`), not literally the word
`unimplemented` -- read the `create:dispatch` line's reason field itself
rather than assuming a fixed string, since it's whatever message the
`VDISPATCH_MATMUL(...)` call site chose.

That's the whole point of this POC: two independently-built `.so`s, loaded
into one process via nothing but `ONEDNN_GPU_PLUGIN_PATH`, each getting a
real shot at oneDNN's own accept/reject convention in listed order -- no
hijacked reference-primitive path, no external validator to keep in sync.

## `matmul_template`

A starting point for writing a new matmul plugin. `plugin_matmul_t`/`pd_t`
have the same shape as a real plugin (see `kf_matmul_plugin` below), but
the kernel-specific parts are left as `TODO`s:

- `pd_t::kernel_supports()` just returns `false` -- a safe, inert default
  that registers correctly and rejects every shape, so everything falls
  through to oneDNN's own matmul implementations until you fill it in with
  your kernel's real accept/reject rule.
- `primitive_t::init()`/`execute()` return `status::unimplemented` --
  fill these in with however your kernel is compiled and submitted (see
  `kf_matmul_plugin/kf_matmul.cpp` for a worked example).

To start a new plugin from it: copy the directory, rename the
target/namespace/struct names, and rename the
`DECLARE_COMMON_PD_T("template:matmul:any", ...)` string so your kernel is
identifiable in `ONEDNN_VERBOSE` output. It builds via the same Building
procedure above, including its `make_archives.sh` step -- even this
template's fully generic, unmodified `pd_t` code (`scales_ok()`, via the
common `attr_scales_ok()` helper every `*_pd.hpp` shares) transitively
touches internal oneDNN symbols with no dynamic-table entry, so the static
archive link is needed here too, not just once your own kernel code reaches
for something internal. See `kf_matmul_plugin`'s section below for the full
explanation; `plugins/common.cmake`'s `plugin_common_setup()` applies it to
every plugin uniformly, so nothing extra is needed in
`matmul_template/CMakeLists.txt` itself.

## `kf_matmul_plugin`

The one complete, working plugin in this directory, and the one to read to
see every part of the mechanism exercised end to end. It dispatches to a
real, evolvable SYCL kernel (`kf_kernel.sycl.hpp`) rather than a stand-in:

- `kf_kernel.sycl.hpp` isn't hand-maintained -- it's mechanically
  regenerated by `update_kernel.sh` from an external kernel-generation
  pipeline's own `kernel.sycl` file (`./update_kernel.sh
  /path/to/kernel.sycl && cmake --build build` to pick up a new kernel
  version). Its evolved body is a plain host-side function that takes a
  SYCL queue and raw USM device pointers and submits its own work --
  `kf_matmul_t::execute()` (`kf_matmul.cpp`) just pulls that queue from
  oneDNN's own stream and the raw pointers from oneDNN's own memory
  storages, and hands them over. This only works when oneDNN's GPU memory
  is USM-backed (benchdnn's default), not buffer-backed -- there's no raw
  pointer to hand a raw-pointer kernel signature otherwise.
- `pd_t::kf_kernel_supports()` (`kf_matmul.cpp`) is hardcoded to exactly
  what that kernel computes today: plain 2D f32 `C = relu(A @ B)`, no bias,
  no scales/zero-points, fixed (non-runtime) dims. Anything outside that
  one shape falls through to oneDNN's own matmul implementations.
- Like every plugin in this directory, this one statically links the needed
  internal oneDNN object code into its own `.so` rather than only dynamic-
  linking against `libdnnl`. oneDNN's build compiles everything with
  `-fvisibility=internal`, so internal C++ symbols a plugin needs (e.g.
  `memory_storage_t` here, or even just the generic `attr_scales_ok()` pd
  helper any matmul-shaped plugin's `scales_ok()` calls) aren't in
  `libdnnl.so`'s dynamic symbol table for `dlopen`/`dlsym` to find --
  confirmed the hard way: this gap isn't specific to `kf_matmul_plugin`'s
  raw-USM-pointer kernel access, since even `matmul_template`, completely
  unmodified, hits it too (via a transitive reference to
  `rnn_create_time_scales_t::set_single_scale`, of all things, pulled in by
  that shared pd helper). `plugins/make_archives.sh` (see Building above)
  works around this by archiving oneDNN's already-built object code so a
  plugin can resolve those symbols at its own link time instead, before
  any dynamic symbol table is involved. `plugins/common.cmake`'s
  `plugin_common_setup()` applies both this archive link and the matching
  `NDEBUG`/`Release` build-type setting to every plugin uniformly, so
  nothing plugin-specific is needed in `kf_matmul_plugin/CMakeLists.txt`
  for this beyond calling `plugin_common_setup()` like any other plugin
  does.

## Registering multiple implementations

`dnnl_gpu_plugin_register(plugin_registrar_t &registrar)` isn't limited to
one `registrar.register_impl(kind, item)` call. A single plugin can call it
several times to register more than one implementation, for the same
primitive kind (e.g. an fp16-specialized and a bf16-specialized variant of
its own kernel) or for different kinds (e.g. matmul and reduction) -- the
registry keeps a separate list of registered implementations per
`primitive_kind_t`, not a single slot.

Multiple plugin `.so`s work the same way: list them colon-separated in
`ONEDNN_GPU_PLUGIN_PATH`, and each is loaded and its
`dnnl_gpu_plugin_register` called in the order listed.

Within a primitive kind, all registered implementations are merged ahead of
oneDNN's own built-in list, in registration order, and
`primitive_desc_iterator_t` walks that merged list from the front, accepting
the first one whose `pd_t::init()` returns success. So if two implementations
both accept a given shape/dtype, whichever was registered first always wins
-- there's no priority or "pick the faster one" mechanism beyond registration
order.
