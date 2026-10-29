#pragma once

// The mesh_launch_k and mesh_post_k kernel entries. mesh_launch_k
// (gfx_mesh_launch_arg_t): a task chunk's windows from the mesh workgroups its
// scan counted. mesh_post_k (gfx_mesh_post_arg_t): a window's live mesh
// primitives, compacted per workgroup, into the dense list setup reads, each
// workgroup's run at the offset the scan of their counts gave it. A group of
// `lanes` threads copies one workgroup's run. Device-only (pulls vx_spawn2.h).

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>
#include <gfx_mesh_abi.h>

__kernel void mesh_launch_k(gfx_mesh_launch_arg_t* __UNIFORM__ arg) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  auto meta = reinterpret_cast<volatile gfx_mesh_meta_t*>(arg->meta);
  if (arg->record) {
    const auto rec = reinterpret_cast<const volatile uint32_t*>(arg->record);
    const bool live = !arg->draw_count ||
                      arg->draw_index < *reinterpret_cast<const volatile uint32_t*>(arg->draw_count);
    uint64_t total = live ? 1u : 0u;
    for (uint32_t k = 0; k < 3; ++k) {
      const uint32_t g = live ? rec[k] : 0u;
      meta->grid[k] = g;
      total *= g;
    }
    // The limit maxMeshWorkGroupTotalCount sets; a larger grid is clamped.
    const uint32_t n = total < GFX_MESH_MAX_WORKGROUPS ? (uint32_t)total : GFX_MESH_MAX_WORKGROUPS;
    meta->meshes = n;
    *reinterpret_cast<uint32_t*>(arg->overflow) = (uint64_t)n > (uint64_t)arg->nwin * arg->window;
  }
  const uint32_t n = meta->meshes;
  auto win = reinterpret_cast<gfx_mesh_window_t*>(arg->windows);
  for (uint32_t w = 0; w < arg->nwin; ++w) {
    const uint64_t f0 = (uint64_t)w * arg->window;
    const uint32_t left = n > f0 ? (uint32_t)(n - f0) : 0u;
    const uint32_t cnt = left < arg->window ? left : arg->window;
    win[w].grid = cnt < arg->grid_max ? cnt : arg->grid_max;
    win[w].prims = cnt * arg->static_prims;
    win[w].count = cnt;
  }
}

__kernel void mesh_post_k(gfx_mesh_post_arg_t* __UNIFORM__ arg) {
  const uint32_t L = arg->lanes;
  const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;
  const auto pcnt = reinterpret_cast<const uint32_t*>(arg->pcnt);
  const auto pbase = reinterpret_cast<const uint32_t*>(arg->pbase);
  const auto live = reinterpret_cast<const pipe_list_entry_t*>(arg->live);
  auto list = reinterpret_cast<pipe_list_entry_t*>(arg->list);
  uint32_t count = arg->count;
  if (arg->count_addr) {
    const uint32_t c = *reinterpret_cast<const volatile uint32_t*>(arg->count_addr);
    count = c < count ? c : count;
  }
  for (uint32_t i = g; i < count * L; i += stride) {
    const uint32_t w = i / L;
    const uint32_t n = pcnt[w], base = pbase[w];
    for (uint32_t k = i % L; k < n; k += L) {
      list[base + k] = live[(size_t)w * arg->pmax + k];
    }
  }
}
