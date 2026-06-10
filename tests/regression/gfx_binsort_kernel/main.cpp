// gfx_v2 on-device bin-sort binning — SimX validation host.
// Generates primitive bboxes, computes the host bin-sort reference, runs the
// SIMT binning kernel, and checks the device output (headers + sorted pids)
// matches the reference exactly.

#include <vortex.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <vector>
#include "common.h"

#define CHECK(expr)                                                      \
  do {                                                                   \
    int _r = (expr);                                                     \
    if (_r != 0) {                                                       \
      std::printf("Error: '%s' returned %d (%s:%d)\n", #expr, _r,        \
                  __FILE__, __LINE__);                                   \
      return -1;                                                         \
    }                                                                    \
  } while (0)

static uint32_t g_num_prims = 200;
static const char* g_kernel_file = "kernel.vxbin";

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
      case 'n': g_num_prims = std::atoi(optarg); break;
      case 'k': g_kernel_file = optarg; break;
      default: break;
    }
  }
}

// Host bin-sort reference (matches the kernel's six stages exactly).
struct RefOut {
  uint32_t P = 0, nbins = 0;
  std::vector<binsort_header_t> headers;
  std::vector<uint32_t> pids;
};

// Host triangle setup — MUST match the device kernel (stage 0) exactly.
static int h_imin3(int a, int b, int c) { int m = a < b ? a : b; return m < c ? m : c; }
static int h_imax3(int a, int b, int c) { int m = a > b ? a : b; return m > c ? m : c; }
static int h_iclamp(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

static binsort_prim_t host_setup_tri(binsort_vertex_t a, binsort_vertex_t b, binsort_vertex_t c) {
  int det = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  binsort_prim_t p = { 0, 0, 0, 0 };
  if (det != 0) {
    int L = h_iclamp(h_imin3(a.x, b.x, c.x), BINSORT_W), R = h_iclamp(h_imax3(a.x, b.x, c.x), BINSORT_W);
    int T = h_iclamp(h_imin3(a.y, b.y, c.y), BINSORT_H), B = h_iclamp(h_imax3(a.y, b.y, c.y), BINSORT_H);
    if (R > L && B > T) { p.bbL = (uint32_t)L; p.bbR = (uint32_t)R; p.bbT = (uint32_t)T; p.bbB = (uint32_t)B; }
  }
  return p;
}

static RefOut host_binsort(const std::vector<binsort_prim_t>& prims) {
  RefOut r;
  std::vector<uint32_t> keys;
  for (uint32_t i = 0; i < prims.size(); ++i) {
    const auto& p = prims[i];
    if (p.bbR <= p.bbL || p.bbB <= p.bbT) continue;   // culled / empty bbox
    int bL = p.bbL >> BINSORT_BIN_LOG, bR = (p.bbR - 1) >> BINSORT_BIN_LOG;
    int bT = p.bbT >> BINSORT_BIN_LOG, bB = (p.bbB - 1) >> BINSORT_BIN_LOG;
    for (int by = bT; by <= bB; ++by)
      for (int bx = bL; bx <= bR; ++bx)
        keys.push_back(((uint32_t)(by * BINSORT_BIN_COLS + bx) << BINSORT_PRIM_BITS) | i);
  }
  r.P = (uint32_t)keys.size();
  std::sort(keys.begin(), keys.end());
  r.pids.resize(r.P);
  uint32_t i = 0;
  while (i < r.P) {
    uint32_t bin = keys[i] >> BINSORT_PRIM_BITS, start = i;
    while (i < r.P && (keys[i] >> BINSORT_PRIM_BITS) == bin) {
      r.pids[i] = keys[i] & BINSORT_PRIM_MASK;
      ++i;
    }
    binsort_header_t h;
    h.bin_x = (uint16_t)(bin % BINSORT_BIN_COLS);
    h.bin_y = (uint16_t)(bin / BINSORT_BIN_COLS);
    h.pids_offset = start;
    h.pids_count = i - start;
    r.headers.push_back(h);
  }
  r.nbins = (uint32_t)r.headers.size();
  return r;
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  std::srand(50);
  const uint32_t n = g_num_prims;

  // Generate triangles: 3 integer-valued screen-space verts each (center + jitter,
  // some off-screen / degenerate so the setup stage exercises culling).
  std::vector<binsort_vertex_t> verts(3 * n);
  for (uint32_t i = 0; i < n; ++i) {
    int cx = std::rand() % BINSORT_W, cy = std::rand() % BINSORT_H;
    for (int k = 0; k < 3; ++k) {
      int jx = (std::rand() % 401) - 200, jy = (std::rand() % 401) - 200;
      verts[3 * i + k] = { cx + jx, cy + jy };
    }
  }
  // Host setup (matches device stage 0) + reference binning.
  std::vector<binsort_prim_t> host_prims(n);
  for (uint32_t i = 0; i < n; ++i)
    host_prims[i] = host_setup_tri(verts[3 * i], verts[3 * i + 1], verts[3 * i + 2]);
  RefOut ref = host_binsort(host_prims);
  std::printf("gfx_binsort_kernel: n=%u  P=%u  bins=%u\n", n, ref.P, ref.nbins);

  const uint32_t B = BINSORT_NUM_BINS;

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  // One CTA: query a valid device CTA size (T threads) before sizing the
  // per-thread scratch (thist is T×B).
  uint32_t one = 1, grid[1], block[1];
  CHECK(vx_device_max_occupancy_grid(dev, 1, &one, grid, block));
  const uint32_t T = block[0];
  const uint32_t Pcap = ref.P ? ref.P : 1;

  vx_buffer_h verts_buf, prims_buf, count_buf, offset_buf, keys_buf, tsum_buf, thist_buf,
              bincount_buf, binbase_buf, headers_buf, pids_buf, meta_buf;
  CHECK(vx_buffer_create(dev, 3 * n * sizeof(binsort_vertex_t), VX_MEM_READ, &verts_buf));
  CHECK(vx_buffer_create(dev, n * sizeof(binsort_prim_t),   VX_MEM_WRITE, &prims_buf));
  CHECK(vx_buffer_create(dev, n * sizeof(uint32_t),         VX_MEM_WRITE, &count_buf));
  CHECK(vx_buffer_create(dev, (n + 1) * sizeof(uint32_t),   VX_MEM_WRITE, &offset_buf));
  CHECK(vx_buffer_create(dev, Pcap * sizeof(uint32_t),      VX_MEM_WRITE, &keys_buf));
  CHECK(vx_buffer_create(dev, T * sizeof(uint32_t),         VX_MEM_WRITE, &tsum_buf));
  CHECK(vx_buffer_create(dev, T * B * sizeof(uint32_t),     VX_MEM_WRITE, &thist_buf));
  CHECK(vx_buffer_create(dev, B * sizeof(uint32_t),         VX_MEM_WRITE, &bincount_buf));
  CHECK(vx_buffer_create(dev, B * sizeof(uint32_t),         VX_MEM_WRITE, &binbase_buf));
  CHECK(vx_buffer_create(dev, B * sizeof(binsort_header_t), VX_MEM_WRITE, &headers_buf));
  CHECK(vx_buffer_create(dev, Pcap * sizeof(uint32_t),      VX_MEM_WRITE, &pids_buf));
  CHECK(vx_buffer_create(dev, 2 * sizeof(uint32_t),         VX_MEM_WRITE, &meta_buf));

  vx_module_h mod = nullptr;
  vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, g_kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  kernel_arg_t karg{};
  karg.num_prims = n;
  CHECK(vx_buffer_address(verts_buf,    &karg.verts_addr));
  CHECK(vx_buffer_address(prims_buf,    &karg.prims_addr));
  CHECK(vx_buffer_address(count_buf,    &karg.count_addr));
  CHECK(vx_buffer_address(offset_buf,   &karg.offset_addr));
  CHECK(vx_buffer_address(keys_buf,     &karg.keys_addr));
  CHECK(vx_buffer_address(tsum_buf,     &karg.tsum_addr));
  CHECK(vx_buffer_address(thist_buf,    &karg.thist_addr));
  CHECK(vx_buffer_address(bincount_buf, &karg.bincount_addr));
  CHECK(vx_buffer_address(binbase_buf,  &karg.binbase_addr));
  CHECK(vx_buffer_address(headers_buf,  &karg.headers_addr));
  CHECK(vx_buffer_address(pids_buf,     &karg.pids_addr));
  CHECK(vx_buffer_address(meta_buf,     &karg.meta_addr));

  CHECK(vx_enqueue_write(q, verts_buf, 0, verts.data(), 3 * n * sizeof(binsort_vertex_t), 0, nullptr, nullptr));

  // CP-sequenced: 6 chained launches. count(0)/emit(2)/hist(3)/scatter(5) are
  // multi-CTA (grid=G); scan(1)/base(4) are single-CTA (grid=1). hist+scatter
  // are BIN-STRIPED (CTA owns bins bin_id%G==cta). Same block size T. Each
  // launch depends on the prior — the launch-drain is the device barrier.
  const uint32_t NSTAGE = 6;
  const uint32_t G = grid[0];
  karg.bin_stripe = (B + G - 1) / G;          // contiguous bins per CTA
  const uint32_t sgrid[NSTAGE] = { 1, 1, 1, G, 1, G };  // setup/scan/emit/base single-CTA; hist/scatter multi-CTA
  kernel_arg_t kargs[NSTAGE];
  vx_launch_info_t li[NSTAGE];
  vx_event_h ev[NSTAGE] = {};
  for (uint32_t s = 0; s < NSTAGE; ++s) {
    kargs[s] = karg; kargs[s].stage = s;
    li[s] = vx_launch_info_t{};
    li[s].struct_size = sizeof(li[s]);
    li[s].kernel      = kern;
    li[s].args_host   = &kargs[s];
    li[s].args_size   = sizeof(kernel_arg_t);
    li[s].ndim        = 1;
    li[s].grid_dim[0]  = sgrid[s];
    li[s].block_dim[0] = T;
    CHECK(vx_enqueue_launch(q, &li[s], s ? 1 : 0, s ? &ev[s - 1] : nullptr, &ev[s]));
  }

  std::vector<uint32_t> h_meta(2, 0);
  std::vector<binsort_header_t> h_headers(B);
  std::vector<uint32_t> h_pids(ref.P ? ref.P : 1, 0);
  std::vector<binsort_prim_t> h_prims(n);
  vx_event_h last = ev[NSTAGE - 1], ev_m = nullptr, ev_h = nullptr, ev_p = nullptr, ev_pr = nullptr;
  CHECK(vx_enqueue_read(q, h_prims.data(),   prims_buf,   0, n * sizeof(binsort_prim_t),           1, &last, &ev_pr));
  CHECK(vx_enqueue_read(q, h_meta.data(),    meta_buf,    0, 2 * sizeof(uint32_t),                 1, &last, &ev_m));
  CHECK(vx_enqueue_read(q, h_headers.data(), headers_buf, 0, ref.nbins * sizeof(binsort_header_t), 1, &last, &ev_h));
  CHECK(vx_enqueue_read(q, h_pids.data(),    pids_buf,    0, ref.P * sizeof(uint32_t),             1, &last, &ev_p));
  CHECK(vx_event_wait_value(ev_pr, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_m, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_h, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_p, 1, VX_TIMEOUT_INFINITE));

  int errors = 0;
  for (uint32_t i = 0; i < n && errors < 8; ++i) {     // setup: device bboxes == host setup
    const auto& a = h_prims[i]; const auto& b = host_prims[i];
    if (a.bbL != b.bbL || a.bbR != b.bbR || a.bbT != b.bbT || a.bbB != b.bbB) {
      std::printf("*** setup prim[%u] dev{%u,%u,%u,%u} != ref{%u,%u,%u,%u}\n",
                  i, a.bbL, a.bbR, a.bbT, a.bbB, b.bbL, b.bbR, b.bbT, b.bbB);
      ++errors;
    }
  }
  if (h_meta[0] != ref.P)    { std::printf("*** P mismatch: dev=%u ref=%u\n", h_meta[0], ref.P); ++errors; }
  if (h_meta[1] != ref.nbins){ std::printf("*** nbins mismatch: dev=%u ref=%u\n", h_meta[1], ref.nbins); ++errors; }
  for (uint32_t i = 0; i < ref.nbins && errors < 16; ++i) {
    const auto& a = h_headers[i]; const auto& b = ref.headers[i];
    if (a.bin_x != b.bin_x || a.bin_y != b.bin_y || a.pids_offset != b.pids_offset || a.pids_count != b.pids_count) {
      std::printf("*** header[%u] dev{%u,%u,off=%u,cnt=%u} != ref{%u,%u,off=%u,cnt=%u}\n",
                  i, a.bin_x, a.bin_y, a.pids_offset, a.pids_count, b.bin_x, b.bin_y, b.pids_offset, b.pids_count);
      ++errors;
    }
  }
  for (uint32_t i = 0; i < ref.P && errors < 16; ++i)
    if (h_pids[i] != ref.pids[i]) { std::printf("*** pid[%u] dev=%u ref=%u\n", i, h_pids[i], ref.pids[i]); ++errors; }

  vx_event_release(ev_pr); vx_event_release(ev_p); vx_event_release(ev_h); vx_event_release(ev_m);
  for (uint32_t s = 0; s < NSTAGE; ++s) vx_event_release(ev[s]);
  vx_buffer_release(verts_buf);
  vx_buffer_release(prims_buf); vx_buffer_release(count_buf); vx_buffer_release(offset_buf);
  vx_buffer_release(keys_buf); vx_buffer_release(tsum_buf); vx_buffer_release(thist_buf);
  vx_buffer_release(bincount_buf); vx_buffer_release(binbase_buf);
  vx_buffer_release(headers_buf); vx_buffer_release(pids_buf); vx_buffer_release(meta_buf);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_release(dev);

  std::printf("RESULT: %s\n", errors == 0 ? "PASS" : "FAIL");
  return errors == 0 ? 0 : 1;
}
