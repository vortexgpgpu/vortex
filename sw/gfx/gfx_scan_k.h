#pragma once

// Device-wide exclusive prefix sum: the scan_k kernel entry (see
// pipe_scan_arg_t). It orders what many threads produced -- a geometry
// shader's per-invocation primitive counts -- across the whole device, where
// the front end's own scans run on one CTA. Device-only (pulls vx_spawn2.h).

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>
#include <gfx_fs_desc_abi.h>   // GFX_QC_*
#include "gfx_cta_k.h"

namespace gfx_scan {

static inline uint32_t scan_umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

using gfx_cta::query_add;

// Commit a geometry stage's capture: each captured stream writes the primitives
// it made -- the total of its lane, its rank among the scanned `streams` -- that
// every one of its buffers has room for, its buffers' filled sizes advance by
// them, and the stream's written and needed counters take them.
static inline void xfb_commit(const gfx_xfb_desc_t* d, const uint32_t* totals,
                              uint32_t streams) {
  const auto rd = reinterpret_cast<const uint32_t*>(d->off_rd);
  const auto wr = reinterpret_cast<uint32_t*>(d->off_wr);
  uint32_t w[4] = { 0, 0, 0, 0 };
  for (uint32_t s = 0; s < 4; ++s) {
    if (!((d->streams >> s) & 1u) || !((streams >> s) & 1u)) {
      continue;
    }
    const uint32_t made = totals[__builtin_popcount(streams & ((1u << s) - 1u))];
    w[s] = made;
    for (uint32_t b = 0; b < 4; ++b) {
      if (!((d->mask >> b) & 1u) || d->buffer_stream[b] != s) {
        continue;
      }
      const uint32_t room = rd[b] <= d->size[b] ? d->size[b] - rd[b] : 0u;
      const uint32_t fit = room / (d->per * d->stride[b]);
      w[s] = scan_umin(w[s], fit);
    }
    if (d->query_block) {
      query_add(d->query_block, GFX_QC_WRITTEN(s), w[s]);
      query_add(d->query_block, GFX_QC_NEEDED(s), made);
    }
  }
  for (uint32_t b = 0; b < 4; ++b) {
    wr[b] = ((d->mask >> b) & 1u) ? rd[b] + w[d->buffer_stream[b]] * d->per * d->stride[b]
                                   : rd[b];
  }
}

} // namespace gfx_scan

__kernel void scan_k(pipe_scan_arg_t* __UNIFORM__ arg) {
  using namespace gfx_scan;
  const uint32_t L = arg->lanes;
  const uint32_t n = arg->n_addr
      ? scan_umin(*reinterpret_cast<const volatile uint32_t*>(arg->n_addr), arg->n) : arg->n;
  const uint32_t tid = threadIdx.x, T = blockDim.x, j = blockIdx.x;
  auto in      = reinterpret_cast<const uint32_t*>(arg->in_addr);
  auto out     = reinterpret_cast<uint32_t*>(arg->out_addr);
  auto part    = reinterpret_cast<uint32_t*>(arg->part_addr);
  auto scratch = reinterpret_cast<uint32_t*>(arg->scratch_addr) + (size_t)j * 2u * T;

  switch (arg->stage) {

  case PIPE_SCAN_UP:
  case PIPE_SCAN_DOWN: {
    const uint32_t lo = scan_umin(j * arg->chunk, n), hi = scan_umin(lo + arg->chunk, n);
    const uint32_t sub = (arg->chunk + T - 1) / T;
    const uint32_t a = scan_umin(lo + tid * sub, hi), b = scan_umin(a + sub, hi);
    for (uint32_t l = 0; l < L; ++l) {
      uint32_t s = 0;
      for (uint32_t i = a; i < b; ++i) s += in[(size_t)i * L + l];
      uint32_t total;
      uint32_t acc = gfx_cta::exclusive(scratch, tid, T, s, &total);
      if (arg->stage == PIPE_SCAN_UP) {
        if (tid == 0) part[j * L + l] = total;
        continue;
      }
      acc += part[j * L + l];
      for (uint32_t i = a; i < b; ++i) {
        const uint32_t v = in[(size_t)i * L + l];
        out[(size_t)i * L + l] = acc;
        acc += v;
      }
    }
  } break;

  case PIPE_SCAN_TOP: {
    const uint32_t G = arg->parts;
    const uint32_t sub = (G + T - 1) / T;
    const uint32_t a = scan_umin(tid * sub, G), b = scan_umin(a + sub, G);
    auto totals = reinterpret_cast<uint32_t*>(arg->total_addr);
    uint32_t sum = 0;
    for (uint32_t l = 0; l < L; ++l) {
      uint32_t s = 0;
      for (uint32_t i = a; i < b; ++i) s += part[i * L + l];
      uint32_t total;
      uint32_t acc = gfx_cta::exclusive(scratch, tid, T, s, &total);
      for (uint32_t i = a; i < b; ++i) {
        const uint32_t v = part[i * L + l];
        part[i * L + l] = acc;
        acc += v;
      }
      if (tid == 0) {
        totals[l] = total;
        for (uint32_t q = 0; arg->query_addr && q < 3; ++q) {
          const uint32_t slot = (arg->query_lane[q] >> (8u * l)) & 0xffu;
          if (slot != PIPE_SCAN_QC_NONE) query_add(arg->query_addr, slot, total);
        }
      }
      sum += total;
    }
    if (tid == 0 && arg->query_addr && arg->query_sum != PIPE_SCAN_QC_NONE) {
      query_add(arg->query_addr, arg->query_sum, sum);
    }
    if (tid == 0 && arg->xfb_addr) {
      xfb_commit(reinterpret_cast<const gfx_xfb_desc_t*>(arg->xfb_addr), totals, arg->streams);
    }
  } break;

  }
}
