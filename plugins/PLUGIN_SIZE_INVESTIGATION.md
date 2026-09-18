# Plugin size investigation

Status: investigation snapshot, 2026-09-16

## Executive result

The original standalone plugin link was large because it statically resolved a
broad oneDNN dependency closure, not because the plugin kernel source was
large. The full oneDNN build retained CPU, graph, registry, and unrelated
primitive code through common engine and implementation-list translation units.

The effective reduction was to build a second, plugin-support oneDNN tree with
those dependency families disabled:

```text
ONEDNN_CPU_RUNTIME=NONE
ONEDNN_GPU_RUNTIME=SYCL
ONEDNN_GPU_VENDOR=INTEL
ONEDNN_BUILD_GRAPH=OFF
ONEDNN_BUILD_TESTS=OFF
ONEDNN_BUILD_EXAMPLES=OFF
-ffunction-sections -fdata-sections
```

That build produced plugins of 25,184,816 bytes each after section garbage
collection and stripping, versus approximately 83.7 MB each from the normal
full build. This is a reduction of approximately 69.9%.

The full build remains useful and must not be replaced by the plugin-support
build for general oneDNN consumers. A plugin and the `libdnnl.so` it loads into
must come from matching source/build/compiler configurations.

## Measurements

### Full/reference build

- Full oneDNN build duration inferred from Ninja metadata: approximately
  4,114.8 seconds (68 minutes 34.8 seconds).
- `libdnnl.so.3.14`: 87,169,456 bytes.
- Function/data-section full build `libdnnl.so.3.14`: 87,208,984 bytes.
- Archive members: 928.
- Members retained by the plugin link: 696.
- Plugin sizes before the custom build:
  - `libkf_matmul_plugin.so`: 83,692,488 bytes.
  - `libmatmul_template_plugin.so`: 83,692,664 bytes.
- `--gc-sections` alone removed only about 5% from this broad dependency
  closure. Stripping removed symbol/debug metadata but did not solve the
  dependency problem.

### Plugin-support build

- Build tree: `build-plugin-support`.
- Build duration: 2,610.01 seconds (43 minutes 30 seconds).
- Archive size: 102,838,756 bytes.
- `libdnnl.so.3.14`: 34,007,776 bytes.
- Members retained by the plugin link: 308.
- Only one retained member matched the prior CPU/graph naming filter:
  `graph_gen_index_kernel.cpp.o`.
- Final stripped plugin size: 25,184,816 bytes for each plugin.
- `dnnl_gpu_plugin_register` remained exported after stripping.

The archive itself is not the final plugin size. It is an input to the linker;
only archive members needed by unresolved references are pulled into the
plugin. The important change was reducing the references and implementation
registries in the oneDNN build, not merely deleting arbitrary archive members.

## Why headers and remaining sources are not enough

Building a plugin with `-I` paths into the main source tree gives the compiler
the declarations and inline code needed to compile the plugin. It does not
make every referenced oneDNN definition available to the runtime loader.

The plugin uses native oneDNN internal C++ types and helpers, including
primitive-descriptor bases, memory storage, stream/engine objects, attributes,
and dispatch/registration machinery. Those references are compiled into the
plugin object. oneDNN builds its library with internal visibility, so many of
these symbols are intentionally absent from `libdnnl.so`'s dynamic symbol
table. A plugin that only links `libdnnl.so` can therefore compile and link
against the on-disk library but fail at `dlopen` time with unresolved internal
symbols.

This was reproduced with both plugins:

- `kf_matmul_plugin` directly uses internal memory/stream functionality for
  its raw-USM kernel submission.
- `matmul_template` does not contain a real kernel, but its normal matmul
  `pd_t` path still reaches shared internal helpers such as
  `attr_scales_ok()`, which transitively referenced an internal
  `rnn_create_time_scales_t::set_single_scale` symbol.

Consequently, the current archive is a workaround for symbol ownership: it
packages already-built oneDNN object files so the plugin link can resolve the
internal definitions into the plugin itself. It is not evidence that all
oneDNN code executes at runtime.

## What pulled in the unrelated code

The linker maps showed this representative dependency chain:

```text
plugin object
  -> primitive/cache/verbose internals
  -> graph backend registration
  -> xpu/sycl/engine_factory.cpp.o
  -> common/engine.cpp.o
  -> CPU implementation-list getters
  -> CPU reference/JIT implementations
```

With CPU runtime enabled, the common engine and factory code exposed paths to
CPU engine creation and implementation-list registries. Those registries
enumerate many operations and ISA variants, so archive extraction pulled in
unrelated convolution, pooling, reorder, RNN, eltwise, softmax, and CPU JIT
objects. This is a build/dependency-graph issue, not primarily template
unrolling or expansion in the plugin kernel.

The custom build removed the broad CPU and graph paths before archive
packaging. Function/data sections plus `--gc-sections` then provide useful
fine-grained cleanup, but they are not sufficient when the broad registries
are still reachable.

## Can the static archive be avoided?

Not with the current plugin ABI and visibility model by merely pointing CMake
at more source/header directories. Practical alternatives are:

1. **Build the plugin as part of oneDNN's link**, so it shares the same
   internal objects and visibility domain. This changes the deployment model:
   it is no longer an independently loaded plugin `.so`.
2. **Export a deliberately narrow plugin-support ABI** from `libdnnl.so`
   (or provide a matching support shared library). The plugin would use only
   stable exported handles/functions rather than internal C++ classes. This
   avoids embedding oneDNN internals but requires ABI and ownership design.
3. **Build a plugin-specific minimal oneDNN support library/archive** from
   selected sources and registries. This preserves separate `.so` loading but
   requires maintaining the source/dependency boundary.
4. **Continue the matching minimal build-tree approach**, narrowing it further
   with `ONEDNN_ENABLE_PRIMITIVE=MATMUL`, `ONEDNN_ENABLE_WORKLOAD=INFERENCE`,
   and the actual target GPU ISA. This is the lowest-risk next experiment.

Manually excluding arbitrary `.o` files from the existing full archive is not a
safe substitute: it can remove a transitive definition needed by the plugin
or by the matching runtime, and the resulting failure may only appear during
`dlopen` or dispatch.

## Follow-up experiments

The next size-reduction candidates are:

```text
ONEDNN_BUILD_GRAPH=OFF
ONEDNN_ENABLE_PRIMITIVE=MATMUL
ONEDNN_ENABLE_WORKLOAD=INFERENCE
ONEDNN_ENABLE_PRIMITIVE_GPU_ISA=<actual target ISA>
ONEDNN_ENABLE_GEMM_KERNELS_ISA=NONE   # only if the plugin does not need it
```

Each candidate needs a matching oneDNN rebuild, archive regeneration, plugin
rebuild, size/map comparison, and functional loading test. LTO/ThinLTO is a
secondary experiment because all archived objects would need to be rebuilt
with LTO and SYCL linking becomes more complex.

## Template-plugin dependency trace

The bare `matmul_template` plugin has no kernel body: `init()` and `execute()`
return `status::unimplemented`. It nevertheless is not a small standalone
class. Its one translation unit instantiates the following oneDNN functionality.

### Direct functionality used by the template

| Template code | oneDNN functionality |
|---|---|
| `plugin_registrar_t` and `impl_list_item_t` | Dynamic plugin registration and a callable primitive-descriptor factory |
| `DECLARE_COMMON_PD_T` | PD cloning, PD creation, implementation naming, primitive construction, cache-aware creation |
| `gpu_matmul_pd_t` / `matmul_pd_t` | Matmul descriptors, source/weights/destination descriptors, dimensions, bias/reduce state, primitive attributes |
| `set_default_params()` | Memory-descriptor tag/blocking initialization and `memory_desc_wrapper` |
| `scales_ok()` | Scales/zero-point attribute inspection and inherited matmul quantization validation |
| `init()` validation | Default-format normalization, supported data types, plain-format checks, post-op validation, dimension limits, zero-point checks |
| `VDISPATCH_MATMUL` | Verbose dispatch diagnostics and `primitive_desc_t::info(engine)` construction |
| SYCL primitive base | The base primitive/PD ABI and the SYCL kernel handle type; the template does not currently call kernel creation or submission |

The plugin object’s direct unresolved oneDNN references were only about 15
meaningful families: primitive cache, primitive hashing, verbose/info,
primitive attributes, memory-descriptor initialization/checking, allocation,
the global zero descriptor, and the RNN scale helper reached by shared
attribute code. The rest were C++/SYCL runtime support.

### First snowball: the common PD macro

`DECLARE_COMMON_PD_T` looks like boilerplate, but it emits
`create_primitive()`, which calls `primitive_t::create_primitive_common()`.
That code references:

```text
primitive_cache()
primitive_cache_iface_t::get_or_create()
primitive_hashing::key_t
cache_blob_t
scratchpad registry
primitive construction and PD cloning
```

The template needs this because the host primitive-descriptor iterator must be
able to create and cache an implementation using the normal oneDNN dispatch
contract. Section GC cannot remove the emitted virtual entry points merely
because this template's own `primitive_t::init()` currently returns
`unimplemented`.

The validation path adds another common closure:

```text
VDISPATCH_MATMUL
  -> primitive_desc_t::info(engine)
  -> pd_info_t / verbose formatting

set_default_params / memory_desc_wrapper
  -> memory_desc initialization, sanity checks, and descriptor helpers

scales_ok
  -> matmul_pd_t::attr_scales_ok
  -> primitive attributes and shared quantization/scale support
```

### Second, larger snowball: the GPU engine implementation list

The linker map for a freshly relinked template plugin with
`--gc-sections` retained 308 custom-build archive members. The important
chain was:

```text
plugin registration / PD support
  -> internal GPU engine/vtable definitions
  -> gpu::engine_t virtual list methods
  -> gpu_impl_list_t::get_implementation_list()
  -> switch over every GPU primitive kind
  -> gpu_*_list.cpp for convolution, reorder, reduction, softmax, RNN, ...
```

`src/gpu/gpu_engine.hpp` implements the engine's virtual methods by calling
`gpu_impl_list_t`. `src/gpu/gpu_impl_list.cpp` has one switch containing all
GPU primitive kinds. Even though this template only registers `matmul`, that
translation unit references every `get_*_impl_list()` function. Once the
engine/vtable path is needed inside the plugin's statically linked internal
support, the monolithic list translation unit makes all those list objects
reachable.

The retained custom map visibly contained unrelated members such as
`gpu_convolution_list.cpp.o`, `gpu_reorder_list.cpp.o`,
`gpu_reduction_list.cpp.o`, `gpu_softmax_list.cpp.o`, RNN/GEMM support, and
many Intel GPU kernel-generator objects. This is the concrete tens-of-MB
snowball; it is not caused by the template's `kernel_supports()` body.

### What section-per-function did and did not do

`-ffunction-sections -fdata-sections` plus `--gc-sections` removes individual
unused functions inside reachable translation units. It cannot remove:

- a virtual function required by a retained vtable;
- a whole implementation-list translation unit whose switch references every
  primitive list;
- static initialization and registration data reachable from the linked
  runtime;
- internal support required because the plugin is a native oneDNN
  implementation rather than a narrow C ABI callback.

Therefore the next meaningful reduction is structural: split the GPU
implementation list by primitive, make the plugin registry/engine support
avoid the all-primitive switch, or define a narrow exported plugin ABI. A
MATMUL-only build option can reduce the compiled archive, but it will not
fully solve this particular monolithic-engine/list boundary unless the
corresponding source organization is also split.

## Singleton and cache risk

The static archive approach does create a real correctness/maintenance risk:
some oneDNN implementation state can exist once in the host `libdnnl.so` and
again in each plugin `.so`.

The clearest case is `common/primitive_cache.cpp`. The template's emitted
`DECLARE_COMMON_PD_T` creation path calls `primitive_cache()`, and the
statically linked copy contains its own function-local
`global_primitive_cache()` singleton. The host oneDNN library has another
copy. These caches do not automatically become one shared cache merely
because they have the same C++ source and type names; the plugin build uses
hidden visibility for its internal symbols.

Possible consequences include:

- plugin-created primitive instances being cached separately from host
  primitive instances;
- host cache-capacity settings not reliably controlling the plugin copy;
- duplicate kernel/cache state and memory retention;
- cache keys or cached objects crossing the host/plugin boundary with
  configuration assumptions that are not identical;
- future oneDNN changes making duplicated internal state unsafe even when the
  current source/compiler combination happens to work.

The plugin registry is different. The normal registration call receives a
reference to the host registry object, and the virtual `register_impl()` call
updates the host registry. The plugin should not call its own
`gpu_plugin_registry_t::instance()` in that path. Nevertheless, statically
embedding registry/engine support remains fragile because it duplicates
internal classes and vtables around the host objects.

This is why the current README's functional demo sets:

```text
ONEDNN_PRIMITIVE_CACHE_CAPACITY=0
```

Disabling the cache avoids reusing plugin-owned primitive objects across
dispatch attempts and removes the most obvious duplicate-cache hazard. It is
a mitigation, not proof that the architecture is safe for all workloads.
The current setup is suitable for a same-source/same-compiler POC, but it
should not be treated as a stable production ABI.

The robust fixes are to keep primitive/cache ownership in the host library
through a narrow exported ABI, or to make the plugin a true component of the
same oneDNN link. A plugin-specific support library would also need explicit
rules for which singletons are host-owned versus plugin-owned; simply
archiving more `.o` files increases this risk.

## Goal-oriented direction

The goal is not merely to produce a smaller shared object. The goal is to
iterate on custom SYCL kernels with minimal rebuild overhead, while preserving
oneDNN's normal primitive dispatch behavior (plugin gets first refusal, then
fallback to built-in implementations). The design should therefore optimize
the kernel-development loop rather than force every kernel author to rebuild
oneDNN.

The preferred direction is a host-owned adapter with a lightweight
kernel-plugin ABI:

```text
host libdnnl.so
  owns PDs, attributes, caches, engines, streams, dispatch, and lifetimes
  -> calls plugin support/query callback
  -> calls plugin create/execute callback when the kernel accepts

plugin .so
  owns only kernel code and kernel-specific state
  -> receives opaque host handles and POD descriptor/attribute views
  -> submits custom SYCL work through a host-provided queue/stream interface
```

The plugin-facing header should not include `src/common` or `src/gpu`
implementation headers. It should define an experimental, fork-local
interface containing:

- versioned opaque engine/stream/memory handles;
- immutable matmul descriptor and attribute views;
- device/ISA information needed for kernel selection;
- USM pointer and shape/stride accessors;
- stream submission/event helpers;
- scratchpad and error/status callbacks;
- registration callbacks for support/query, create, execute, and destroy.

This preserves the useful behavior demonstrated by the current POC while
moving ownership of oneDNN state back to the host. A plugin can then be
rebuilt with only its SYCL sources and generated kernel code; no archive
regeneration or oneDNN rebuild is needed for ordinary kernel changes.

### Incremental migration plan

1. Keep the current native C++ plugin API as the reference dispatch POC.
2. Add a fork-local experimental ABI v2 beside it; do not destabilize v1.
3. Implement a host-side matmul adapter that owns the normal oneDNN PD and
   translates its descriptors/attributes into ABI-v2 views.
4. Port `kf_matmul_plugin` to the adapter while keeping its existing SYCL
   kernel body and accept/reject behavior.
5. Compare dispatch, correctness, plugin rebuild time, and `.so` size against
   the current implementation.
6. Once the adapter works, make the minimal plugin-support build optional
   rather than required for kernel iteration.

This direction addresses the actual objective: full oneDNN is built once,
custom kernel authors rebuild only the plugin, and oneDNN remains responsible
for caches, engine/stream lifetime, implementation ordering, and fallback.

## Alternative: incremental static kernel injection

If deployment can assume one monolithic `libdnnl.so`, the simpler alternative
is to inject a selected set of custom SYCL kernel implementations directly
into that oneDNN build:

```text
oneDNN CMake build
  -> custom-kernel object library
  -> normal oneDNN implementation/dispatch lists
  -> one monolithic libdnnl.so
```

This avoids the standalone-plugin ABI and avoids duplicate caches entirely.
The injected implementation can use oneDNN's native internal classes and
registries because it is linked into the same image. With Ninja/CMake
dependency tracking, changing a kernel should compile the changed kernel
objects and relink `libdnnl.so`; it should not require recompiling all of
oneDNN. SYCL device-image compilation and the final shared-library link remain
the expected incremental costs.

A practical injection mechanism would be a CMake option or generated
manifest, for example:

```text
ONEDNN_CUSTOM_GPU_KERNEL_SOURCES=<manifest>
```

The manifest would add an object library and oneDNN-side registration
trampolines/list entries. Kernel developers would still edit and rebuild their
kernel sources, but the resulting implementation would be part of the normal
oneDNN dispatch path.

### Choosing between the two modes

| Mode | Kernel edit rebuild | ABI risk | Deployment |
|---|---|---|---|
| Static injection into monolithic `libdnnl.so` | Changed kernel compile plus oneDNN relink | Lowest | Must distribute/rebuild the host oneDNN library |
| Dynamic plugin with host trampoline ABI | Plugin compile only; no host relink | Requires versioned ABI | Host oneDNN remains unchanged |

Static injection is the best first implementation if the KF workflow controls
the oneDNN build and a host-library relink is acceptable. The dynamic
trampoline ABI is preferable when multiple kernel teams need to ship or
iterate plugins independently of the host oneDNN binary. Both modes can share
the same kernel-facing descriptor/stream/memory view and callback concepts;
only the host registration and linking layer differs.

## Comparison with the current KF Docker evaluation

The current KF evaluation repository is:

```text
https://github.com/intel-sandbox/kernelfoundry.kernel-eval
```

Its oneDNN path does **not** build a native oneDNN plugin implementation.
Instead it:

1. clones and builds oneDNN once in the Docker image;
2. applies a small patch to the generic SYCL reference primitive
   (`patches/onednn/ref_matmul.diff`);
3. builds only the evolving `kernel.sycl` as a small preloadable shared
   library;
4. exposes an `extern "C"` entry point;
5. uses a weak symbol plus `LD_PRELOAD` to route the reference primitive's
   `execute()` to that kernel.

The patched oneDNN reference primitive still owns PD creation, descriptor
validation, memory storage, stream selection, cache behavior, and execution
context. The generated kernel receives a SYCL queue and raw pointers. This is
very close to the host-trampoline direction described above, but implemented
as a task-specific preload hook rather than a general plugin registry.

### What KF can already do well

For the kernel-development loop, KF already has the important properties:

- oneDNN is built once and cached by a commit/patch/options fingerprint;
- editing the evolving kernel does not rebuild oneDNN;
- the kernel artifact is rebuilt independently;
- PyTorch and benchdnn can exercise the same kernel body;
- benchdnn compares against the native optimized oneDNN primitive for
  performance;
- the host supplies the queue and memory pointers.

For this goal, a split plugin is not automatically justified. The current
preload path is smaller, simpler, and avoids the duplicated primitive-cache
problem entirely.

### What the KF preload path cannot do

The preload hook is deliberately narrower than a real oneDNN implementation
plugin:

- it injects into `sycl:ref:any`'s `execute()` path rather than registering a
  separate implementation in the primitive implementation list;
- the reference primitive must accept and initialize the operation before the
  custom kernel runs;
- the guard is task-specific and returns an error for unsupported
  configurations rather than naturally rejecting and falling through to the
  next implementation;
- it inherits reference-primitive constraints such as plain formats,
  work-size limits, reference scratchpad/configuration, and limited
  post-op/layout coverage;
- it cannot provide an independent PD setup, scratchpad policy, format
  selection, or implementation-specific primitive lifecycle;
- it is currently one operation entry point per patched primitive rather than
  a general set of independently registered implementations.

The KF README explicitly describes these limitations and distinguishes custom
runtime measurements through the patched reference path from optimized native
oneDNN baselines.

### Decision implication

There are two different goals:

| Goal | Best current mechanism |
|---|---|
| Evolve and benchmark custom SYCL kernels quickly | KF-style host hook/preload |
| Ship a first-class implementation with natural accept/reject and fallback | Host-owned trampoline/adapter or static oneDNN injection |

Therefore the immediate kernel-evaluation workflow should likely remain KF-like:
keep oneDNN patched once, load/rebuild only the kernel artifact, and avoid
embedding native oneDNN C++ in that artifact. The plugin split is justified
only if the project specifically needs first-class implementation-list
semantics, multiple independently registered kernels, or production dispatch
outside the constrained reference path.

## The actual boundary we want

The desired execution flow is reasonable:

```text
oneDNN builds implementation lists
  -> asks registered plugin implementation whether it applies
  -> plugin uses oneDNN-derived operation properties to decide
  -> oneDNN creates the selected primitive
  -> execution uses oneDNN stream/memory objects plus the custom SYCL kernel
```

The current POC crosses that boundary too low in the stack. Its plugin
`pd_t` is itself a native oneDNN `primitive_desc_t` subclass and its
`primitive_t` is itself a native oneDNN primitive subclass. That is why
seemingly modest reuse of:

- PD property extraction;
- memory descriptor and format utilities;
- attribute/post-op inspection;
- stream and memory access;
- primitive setup and cache integration;

also references the native engine vtable, primitive cache, verbose machinery,
and monolithic implementation lists. The tens-of-MB snowball is the price of
reusing the *ownership and dispatch classes*, not the price of the helper
algorithms themselves.

The desired boundary should instead be:

```text
host oneDNN adapter
  owns native PD/primitive, cache, engine, stream, memory, and dispatch
  extracts normalized operation properties
  performs or delegates applicability checks
  calls the plugin kernel callback

plugin
  receives a compact descriptor/attribute view
  may reuse helper logic represented by that view
  owns kernel compilation/configuration state
  submits the custom SYCL kernel through the host-provided execution view
```

This preserves the useful oneDNN behavior without asking the plugin to embed
oneDNN's implementation framework. The reusable surface should be deliberately
small and data-oriented: dimensions, data types, strides/tags, post-op
summary, scales/zero-points summary, device capabilities, USM pointers, and
stream submission. It should not expose `primitive_desc_t`, `primitive_t`,
`engine_t`, `memory_storage_t`, or the global primitive cache to the plugin.

For static injection, the same boundary can be implemented with native C++
objects because everything is in one image; the plugin/kernel source can then
use native helper functions without duplicating them. For dynamic plugins,
the adapter must translate the same information into opaque handles/POD views.

## Durable build locations

- Full/reference build: `build-full-reference`
- Plugin-support build: `build-plugin-support`
- Archive generation: `plugins/make_archives.sh`
- Shared plugin setup: `plugins/common.cmake`

Generated plugin build directories and archives are intentionally not part of
this source note.
