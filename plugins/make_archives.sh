#!/usr/bin/env bash
set -euo pipefail

# Shared across all plugins in this directory -- not specific to any one
# plugin. ONEDNN_BUILD_DIR must match what's passed to
# `cmake -DONEDNN_BUILD_DIR=...` for the plugin itself: this is the build
# tree whose already-compiled object code gets archived. Output lands under
# that same tree (ONEDNN_BUILD_DIR/archives) so there's one path to keep in
# sync, not two: a plugin's own CMakeLists.txt (e.g.
# kf_matmul_plugin/CMakeLists.txt's KF_ONEDNN_ARCHIVE_DIR) derives its
# archive-directory default from ONEDNN_BUILD_DIR the same way.
B="${1:?Usage: make_archives.sh <ONEDNN_BUILD_DIR>}"
OUT="$B/archives"
mkdir -p "$OUT"

# oneDNN's CPU engine holds static impl-list registries that enumerate every
# implementation for every op (all ISA variants of every convolution/pooling/
# etc.), so anything that references CPU engine creation transitively drags
# in the whole CPU implementation list -- confirmed: archiving just
# common+cpu+gpu+xpu+graph_utils still pulled in ~100s of unrelated
# dnnl::impl::cpu::x64::* convolution/JIT symbols once archive-member pulling
# started working correctly. Rather than chase that boundary file by file,
# archive literally ALL already-built object code under src/ (every .dir
# CMake object directory) into one comprehensive archive -- this is the same
# object code libdnnl.so itself is made of, just packaged as an archive
# instead of linked into a .so, so by definition nothing needed is missing.
# No recompilation: reuses .o files the main SHARED oneDNN build already
# produced.
rm -f "$OUT/libdnnl_full.a"
find "$B/src" -name '*.o' -print0 | xargs -0 ar rcs "$OUT/libdnnl_full.a"
ls -la "$OUT"
