#pragma once

// Task and mesh stage kernels (gfx_mesh_abi.h): the persistent grids that walk a
// draw's task and mesh workgroups, one workgroup per CTA iteration, through the
// translated shader's body, and a mesh workgroup's compaction of its live
// primitives. Linked into each translated task or mesh shader (gfx_task_k.cpp,
// gfx_mesh_k.cpp). Device-only (pulls vx_spawn2.h).
//
// The loop bounds come from the arguments and the draw's counts in memory, the
// same for every thread of a CTA, so a CTA runs every iteration whole and the
// barriers in the shader and here stay matched.

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>   // pipe_list_entry_t
#include <gfx_fs_desc_abi.h>    // GFX_QC_*
#include <gfx_mesh_abi.h>
#include "gfx_cta_k.h"

extern "C" void vp_task_main(const gfx_mesh_wg_t* wg);
extern "C" void vp_mesh_main(const gfx_mesh_wg_t* wg);

namespace gfx_mesh {

static inline uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Workgroup i of a grid as gl_WorkGroupID, x varying fastest.
static inline void wg_coords(uint32_t i, const uint32_t* grid, uint32_t* id) {
  const uint32_t gx = grid[0] ? grid[0] : 1u;
  const uint32_t gy = grid[1] ? grid[1] : 1u;
  id[0] = i % gx;
  i /= gx;
  id[1] = i % gy;
  id[2] = i / gy;
}

// The task that launched mesh workgroup w: the last one whose first workgroup
// is at or below w, which skips the tasks that launched none.
static inline uint32_t task_of(const uint32_t* base, uint32_t tasks, uint32_t w) {
  uint32_t lo = 0, hi = tasks;
  while (hi - lo > 1) {
    const uint32_t mid = (lo + hi) / 2;
    if (base[mid] <= w) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

static inline void task_stage(const gfx_task_arg_t* arg) {
  const auto meta = reinterpret_cast<const gfx_mesh_meta_t*>(arg->meta);
  const uint32_t n = meta->tasks;
  const uint32_t G = gridDim.x;
  gfx_mesh_wg_t wg{};
  wg.desc = arg->desc;
  for (uint32_t k = 0; k < 3; ++k) {
    wg.num_wg[k] = meta->grid[k];
  }
  wg.draw_id = arg->draw_id;
  wg.view = arg->view;
  // The chunk's invocations, counted once.
  if (arg->query && blockIdx.x == 0 && threadIdx.x == 0) {
    gfx_cta::query_add(arg->query, GFX_QC_TASK_INVOCATIONS, (uint64_t)n * arg->block);
  }
  for (uint32_t t = blockIdx.x; t < n; t += G) {
    wg_coords(arg->first + t, meta->grid, wg.wg_id);
    wg.payload = arg->payload + (uint64_t)t * arg->payload_stride;
    wg.grid = arg->grid + 16ull * t;
    wg.count = arg->count + 4ull * t;
    vp_task_main(&wg);
    // The next workgroup starts with its own workgroup memory.
    if (arg->flags & GFX_MESH_F_SHARED) {
      __syncthreads();
    }
  }
}

// A staged primitive survives when it is not culled and names vertices the
// workgroup output; any other index is undefined, and the primitive is dropped.
static inline bool prim_live(const pipe_list_entry_t& e, uint32_t verts, uint32_t pv,
                             bool cull) {
  if (cull && e.aux) {
    return false;
  }
  for (uint32_t c = 0; c < pv; ++c) {
    if (e.rec[c] >= verts) {
      return false;
    }
  }
  return true;
}

// Mesh workgroup wl of the window, after its body: its live primitives, in
// index order, to its slot of `live` with their records named across the
// window, and their count to pcnt[wl]. Threads take contiguous runs of
// primitives and a CTA scan places each run. Leaves the counts zeroed for the
// next workgroup.
static inline void mesh_compact(const gfx_mesh_arg_t* arg, uint32_t wl, uint32_t* counts,
                                uint32_t* scratch, uint32_t tid, uint32_t T) {
  const uint32_t verts = umin(counts[0], arg->vmax);
  const uint32_t prims = umin(counts[1], arg->pmax);
  const uint32_t pv = arg->prim_verts;
  const bool cull = (arg->flags & GFX_MESH_F_CULL) != 0;
  const auto stage = reinterpret_cast<const pipe_list_entry_t*>(arg->prims)
                   + (size_t)wl * arg->pmax;
  auto out = reinterpret_cast<pipe_list_entry_t*>(arg->live) + (size_t)wl * arg->pmax;
  const uint32_t chunk = (prims + T - 1) / T;
  const uint32_t lo = umin(tid * chunk, prims), hi = umin(lo + chunk, prims);
  uint32_t n = 0;
  for (uint32_t k = lo; k < hi; ++k) {
    n += prim_live(stage[k], verts, pv, cull) ? 1u : 0u;
  }
  uint32_t total;
  uint32_t at = gfx_cta::exclusive(scratch, tid, T, n, &total);
  if (tid == 0) {
    counts[0] = 0;
    counts[1] = 0;
  }
  const uint32_t rec0 = wl * arg->vmax;
  const uint32_t prim0 = wl * arg->pmax;
  const bool primrec = (arg->flags & GFX_MESH_F_PRIMREC) != 0;
  for (uint32_t k = lo; k < hi; ++k) {
    const pipe_list_entry_t e = stage[k];
    if (!prim_live(e, verts, pv, cull)) {
      continue;
    }
    pipe_list_entry_t o;
    for (uint32_t c = 0; c < 3; ++c) {
      o.rec[c] = rec0 + e.rec[c < pv ? c : pv - 1];
    }
    o.aux = primrec ? ((prim0 + k) & PIPE_LIST_AUX_MASK) : 0u;
    out[at++] = o;
  }
  if (tid == 0) {
    reinterpret_cast<uint32_t*>(arg->pcnt)[wl] = total;
  }
}

static inline void mesh_stage(const gfx_mesh_arg_t* arg) {
  const auto meta = reinterpret_cast<const gfx_mesh_meta_t*>(arg->meta);
  const uint32_t tid = threadIdx.x;
  const uint32_t T = blockDim.x;
  const uint32_t G = gridDim.x;
  const uint32_t end = umin(arg->first + arg->count, meta->meshes);
  const bool stat = (arg->flags & GFX_MESH_F_STATIC) != 0;
  const uint32_t S = arg->static_prims;
  const bool task = (arg->flags & GFX_MESH_F_TASK) != 0;
  auto lmem = reinterpret_cast<uint8_t*>(__local_mem()) + arg->lmem;
  auto counts = reinterpret_cast<uint32_t*>(lmem);
  auto scratch = counts + 4;

  // The draw's invocations, and with static counts its primitives, are
  // counted once, by the first window.
  if (arg->query && arg->first == 0 && blockIdx.x == 0 && tid == 0) {
    gfx_cta::query_add(arg->query, GFX_QC_MESH_INVOCATIONS, (uint64_t)meta->meshes * arg->block);
    if (stat) {
      const uint64_t made = (uint64_t)meta->meshes * S;
      gfx_cta::query_add(arg->query, GFX_QC_PRIMS_GENERATED, made);
      gfx_cta::query_add(arg->query, GFX_QC_CLIP_INVOCATIONS, made);
      gfx_cta::query_add(arg->query, GFX_QC_CLIP_PRIMITIVES, made);
    }
  }
  if (!stat) {
    if (tid == 0) {
      counts[0] = 0;
      counts[1] = 0;
    }
    __syncthreads();
  }

  gfx_mesh_wg_t wg{};
  wg.desc = arg->desc;
  wg.counts = reinterpret_cast<uint64_t>(counts);
  wg.draw_id = arg->draw_id;
  wg.view = arg->view;
  for (uint32_t w = arg->first + blockIdx.x; w < end; w += G) {
    const uint32_t wl = w - arg->first;
    uint32_t t = 0, local = w;
    const uint32_t* grid = meta->grid;
    if (task && arg->fanout) {
      t = w / arg->fanout;
      local = w - t * arg->fanout;
      grid = arg->fan_grid;
    } else if (task) {
      const auto base = reinterpret_cast<const uint32_t*>(arg->task_base);
      t = task_of(base, meta->tasks, w);
      local = w - base[t];
      grid = reinterpret_cast<const uint32_t*>(arg->task_grid) + 4u * t;
    }
    wg_coords(local, grid, wg.wg_id);
    for (uint32_t k = 0; k < 3; ++k) {
      wg.num_wg[k] = grid[k];
    }
    wg.payload = task ? arg->payload + (uint64_t)t * arg->payload_stride : 0u;
    wg.vrec = arg->vrec + (uint64_t)wl * arg->vmax * arg->vstride;
    wg.vrec_index = wl * arg->vmax;
    wg.prec = arg->prec + (uint64_t)wl * arg->pmax * arg->pstride;
    auto entries = reinterpret_cast<pipe_list_entry_t*>(arg->prims)
                 + (size_t)wl * (stat ? S : arg->pmax);
    wg.prims = reinterpret_cast<uint64_t>(entries);

    if (stat) {
      // Every entry starts on the workgroup's vertex 0, so a primitive whose
      // indices the body leaves unwritten is degenerate and reads only this
      // workgroup's records; beside the indices, its primitive record.
      const bool primrec = (arg->flags & GFX_MESH_F_PRIMREC) != 0;
      for (uint32_t k = tid; k < S; k += T) {
        pipe_list_entry_t e;
        e.rec[0] = e.rec[1] = e.rec[2] = wg.vrec_index;
        e.aux = primrec ? ((wl * arg->pmax + k) & PIPE_LIST_AUX_MASK) : 0u;
        entries[k] = e;
      }
      // The barrier also keeps the next workgroup's body from starting on
      // workgroup memory this one still uses.
      __syncthreads();
      vp_mesh_main(&wg);
      continue;
    }

    // A primitive is live until the body culls it; the slot may hold the flags
    // of a workgroup an earlier window ran.
    if (arg->flags & GFX_MESH_F_CULL) {
      for (uint32_t k = tid; k < arg->pmax; k += T) {
        entries[k].aux = 0;
      }
      __syncthreads();
    }
    vp_mesh_main(&wg);
    __syncthreads();
    mesh_compact(arg, wl, counts, scratch, tid, T);
    __syncthreads();
  }
}

} // namespace gfx_mesh
