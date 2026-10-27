#pragma once

// Input assembly with primitive restart -- device kernel stages (ia_k).
//
// A restart index cuts a draw's strips, fans and lists into runs, so where each
// primitive starts depends on every restart before it. ia_k finds that with a
// segmented scan over the draw's index positions, read in place from the
// application's index buffer, and writes the primitives in draw order to the
// draw's gfx_ia_prim_t map, which the vertex kernel then assembles from. The
// count it finds becomes the draw's gfx_ia_t::prims. Device-only.

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>
#include <gfx_fs_desc_abi.h>   // GFX_QC_*

namespace gfx_ia {

#define IA_NONE 0xffffffffu

// The index at draw position `pos`, as read: 0 past the bound buffer.
static inline uint32_t ia_index(const gfx_ia_t* ia, uint32_t pos) {
  const uint64_t i = (uint64_t)ia->first + pos;
  if (i >= ia->index_limit) {
    return 0;
  }
  switch (ia->index_size) {
  case 1:
    return reinterpret_cast<const uint8_t*>(ia->index_addr)[i];
  case 2:
    return reinterpret_cast<const uint16_t*>(ia->index_addr)[i];
  default:
    return reinterpret_cast<const uint32_t*>(ia->index_addr)[i];
  }
}

static inline bool ia_is_restart(const gfx_ia_t* ia, uint32_t pos) {
  return ia_index(ia, pos) == ia->restart_index;
}

// Whether position `pos` of the run that begins at `run` ends a primitive.
static inline bool ia_ends(const gfx_ia_t* ia, uint32_t pos, uint32_t run) {
  const uint32_t in_run = pos - run;
  if (ia->topology == GFX_IA_LIST) {
    return (in_run + 1) % ia->per == 0;
  }
  if (ia->topology == GFX_IA_TRI_STRIP_ADJ) {
    return in_run >= 5 && (in_run & 1u);
  }
  return in_run + 1 >= ia->per;
}

// The positions of [a, b) that end a primitive of the run beginning at `run`
// (run <= a), when none of them is a restart.
static inline uint32_t ia_ends_in(const gfx_ia_t* ia, uint32_t a, uint32_t b, uint32_t run) {
  if (a >= b) {
    return 0;
  }
  if (ia->topology == GFX_IA_LIST) {
    const uint32_t first = a + (ia->per - 1 - (a - run) % ia->per);
    return first < b ? (b - 1 - first) / ia->per + 1 : 0;
  }
  if (ia->topology == GFX_IA_TRI_STRIP_ADJ) {
    const uint32_t lo = a - run > 5 ? a - run : 5;
    const uint32_t first = lo | 1u;
    return first < b - run ? (b - run - 1 - first) / 2 + 1 : 0;
  }
  const uint32_t first = run + ia->per - 1 > a ? run + ia->per - 1 : a;
  return first < b ? b - first : 0;
}

// Add `n` to 64-bit counter `slot` of a GFX_QC_* block with 32-bit atomics, the
// way every other stage adds to it.
static inline void ia_count(uint64_t block, uint32_t slot, uint64_t n) {
  auto w = reinterpret_cast<uint32_t*>(block + slot * 8u);
  const uint32_t lo = (uint32_t)n;
  const uint32_t old = __atomic_fetch_add(&w[0], lo, __ATOMIC_RELAXED);
  const uint32_t hi = (uint32_t)(n >> 32) + (old + lo < old ? 1u : 0u);
  if (hi) {
    __atomic_fetch_add(&w[1], hi, __ATOMIC_RELAXED);
  }
}

// COUNT for thread `gid`: its chunk's first and last restart, and the
// primitives that end after its first restart, which depend on nothing before
// the chunk.
static inline void ia_stage_count(const gfx_ia_t* ia, uint32_t* part, uint32_t gid, uint32_t N,
                                  uint32_t count, uint32_t chunk) {
  if (gid >= N) {
    return;
  }
  const uint32_t lo = gid * chunk < count ? gid * chunk : count;
  const uint32_t hi = lo + chunk < count ? lo + chunk : count;
  uint32_t first = hi, last = IA_NONE, ends = 0, run = 0;
  for (uint32_t pos = lo; pos < hi; ++pos) {
    if (ia_is_restart(ia, pos)) {
      if (first == hi) {
        first = pos;
      }
      last = pos;
      run = pos + 1;
    } else if (first != hi && ia_ends(ia, pos, run)) {
      ++ends;
    }
  }
  part[3 * gid]     = first;
  part[3 * gid + 1] = last;
  part[3 * gid + 2] = ends;
}

// SCAN on one CTA: thread t takes a contiguous group of the COUNT chunks. A
// chunk's run on entry is one past the last restart of every chunk before it,
// and its primitives before its own first restart follow from that in closed
// form. The chunk's partials become its run on entry and its first primitive.
// Each barrier follows a fence: another warp reads what this one stored, and a
// barrier alone does not wait for the stores to land.
static inline void ia_stage_scan(gfx_ia_t* ia, uint32_t* part, uint32_t* tsum, uint32_t N,
                                 uint32_t count, uint32_t chunk) {
  const uint32_t tid = threadIdx.x;
  const uint32_t T = blockDim.x;
  const uint32_t group = (N + T - 1) / T;
  const uint32_t glo = tid * group < N ? tid * group : N;
  const uint32_t ghi = glo + group < N ? glo + group : N;
  uint32_t* trun = tsum;
  uint32_t* tcnt = tsum + T;
  {
    uint32_t last = IA_NONE;
    for (uint32_t i = glo; i < ghi; ++i) {
      if (part[3 * i + 1] != IA_NONE) {
        last = part[3 * i + 1];
      }
    }
    trun[tid] = last;
  }
  vx_fence();
  __syncthreads();
  if (tid == 0) {
    uint32_t run = 0;
    for (uint32_t t = 0; t < T; ++t) {
      const uint32_t last = trun[t];
      trun[t] = run;
      if (last != IA_NONE) {
        run = last + 1;
      }
    }
  }
  vx_fence();
  __syncthreads();
  {
    uint32_t run = trun[tid], acc = 0;
    for (uint32_t i = glo; i < ghi; ++i) {
      const uint32_t lo = i * chunk < count ? i * chunk : count;
      const uint32_t first = part[3 * i], last = part[3 * i + 1];
      const uint32_t n = ia_ends_in(ia, lo, first, run) + part[3 * i + 2];
      part[3 * i] = run;
      part[3 * i + 2] = acc;
      acc += n;
      if (last != IA_NONE) {
        run = last + 1;
      }
    }
    tcnt[tid] = acc;
  }
  vx_fence();
  __syncthreads();
  if (tid == 0) {
    uint32_t acc = 0;
    for (uint32_t t = 0; t < T; ++t) {
      const uint32_t n = tcnt[t];
      tcnt[t] = acc;
      acc += n;
    }
    ia->prims = acc;
    if (ia->query_block) {
      const uint64_t n = (uint64_t)acc * ia->query_scale;
      ia_count(ia->query_block, GFX_QC_IA_PRIMITIVES, n);
      if (ia->gs_invocations) {
        ia_count(ia->query_block, GFX_QC_GS_INVOCATIONS, n * ia->gs_invocations);
      } else {
        ia_count(ia->query_block, GFX_QC_PRIMS_GENERATED, n);
        ia_count(ia->query_block, GFX_QC_CLIP_INVOCATIONS, n);
        ia_count(ia->query_block, GFX_QC_CLIP_PRIMITIVES, n);
        if (ia->query_xfb) {
          ia_count(ia->query_block, GFX_QC_XFB_NEEDED, n);
        }
      }
    }
  }
  vx_fence();
  __syncthreads();
  for (uint32_t i = glo; i < ghi; ++i) {
    part[3 * i + 2] += tcnt[tid];
  }
}

// EMIT for thread `gid`: its chunk's primitives to the map, from the run and
// first primitive SCAN gave it.
static inline void ia_stage_emit(const gfx_ia_t* ia, const uint32_t* part, uint32_t gid,
                                 uint32_t N, uint32_t count, uint32_t chunk) {
  if (gid >= N) {
    return;
  }
  auto map = reinterpret_cast<gfx_ia_prim_t*>(ia->map_addr);
  const uint32_t lo = gid * chunk < count ? gid * chunk : count;
  const uint32_t hi = lo + chunk < count ? lo + chunk : count;
  uint32_t run = part[3 * gid], w = part[3 * gid + 2];
  for (uint32_t pos = lo; pos < hi; ++pos) {
    if (ia_is_restart(ia, pos)) {
      run = pos + 1;
    } else if (ia_ends(ia, pos, run)) {
      map[w++] = gfx_ia_prim_t{ pos, run };
    }
  }
}

} // namespace gfx_ia

__kernel void ia_k(ia_arg_t* __UNIFORM__ arg) {
  using namespace gfx_ia;
  auto ia   = reinterpret_cast<gfx_ia_t*>(arg->ia_addr);
  auto part = reinterpret_cast<uint32_t*>(arg->part_addr);
  auto tsum = reinterpret_cast<uint32_t*>(arg->tsum_addr);
  const uint32_t N = arg->threads;
  const uint32_t count = ia->count;
  const uint32_t chunk = (count + N - 1) / N;
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  switch (arg->stage) {
  case IA_STAGE_COUNT:
    ia_stage_count(ia, part, gid, N, count, chunk);
    break;
  case IA_STAGE_SCAN:
    ia_stage_scan(ia, part, tsum, N, count, chunk);
    break;
  case IA_STAGE_EMIT:
    ia_stage_emit(ia, part, gid, N, count, chunk);
    break;
  case IA_STAGE_ALL:
    ia_stage_count(ia, part, gid, N, count, chunk);
    vx_fence();
    __syncthreads();
    ia_stage_scan(ia, part, tsum, N, count, chunk);
    vx_fence();
    __syncthreads();
    ia_stage_emit(ia, part, gid, N, count, chunk);
    break;
  }
}
