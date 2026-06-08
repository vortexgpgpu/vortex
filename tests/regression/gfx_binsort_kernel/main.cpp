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

static RefOut host_binsort(const std::vector<binsort_prim_t>& prims) {
  RefOut r;
  std::vector<uint32_t> keys;
  for (uint32_t i = 0; i < prims.size(); ++i) {
    const auto& p = prims[i];
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

  // Generate primitive bboxes (clamped to the render target).
  std::vector<binsort_prim_t> prims(n);
  for (uint32_t i = 0; i < n; ++i) {
    int cx = std::rand() % BINSORT_W, cy = std::rand() % BINSORT_W;
    int w = 1 + std::rand() % 200, h = 1 + std::rand() % 200;
    int L = std::max(0, cx - w), R = std::min(BINSORT_W, cx + w);
    int T = std::max(0, cy - h), B = std::min(BINSORT_W, cy + h);
    if (R <= L) R = L + 1;
    if (B <= T) B = T + 1;
    prims[i] = { (uint32_t)L, (uint32_t)R, (uint32_t)T, (uint32_t)B };
  }

  RefOut ref = host_binsort(prims);
  std::printf("gfx_binsort_kernel: n=%u  P=%u  bins=%u\n", n, ref.P, ref.nbins);

  const uint32_t max_bins = BINSORT_BIN_COLS * BINSORT_BIN_COLS;

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  vx_buffer_h prims_buf, count_buf, offset_buf, keys_buf, headers_buf, pids_buf, meta_buf;
  CHECK(vx_buffer_create(dev, n * sizeof(binsort_prim_t),        VX_MEM_READ,  &prims_buf));
  CHECK(vx_buffer_create(dev, n * sizeof(uint32_t),             VX_MEM_WRITE, &count_buf));
  CHECK(vx_buffer_create(dev, (n + 1) * sizeof(uint32_t),       VX_MEM_WRITE, &offset_buf));
  CHECK(vx_buffer_create(dev, (ref.P ? ref.P : 1) * sizeof(uint32_t),       VX_MEM_WRITE, &keys_buf));
  CHECK(vx_buffer_create(dev, max_bins * sizeof(binsort_header_t),          VX_MEM_WRITE, &headers_buf));
  CHECK(vx_buffer_create(dev, (ref.P ? ref.P : 1) * sizeof(uint32_t),       VX_MEM_WRITE, &pids_buf));
  CHECK(vx_buffer_create(dev, 2 * sizeof(uint32_t),            VX_MEM_WRITE, &meta_buf));

  vx_module_h mod = nullptr;
  vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, g_kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  kernel_arg_t karg{};
  karg.num_prims = n;
  CHECK(vx_buffer_address(prims_buf,   &karg.prims_addr));
  CHECK(vx_buffer_address(count_buf,   &karg.count_addr));
  CHECK(vx_buffer_address(offset_buf,  &karg.offset_addr));
  CHECK(vx_buffer_address(keys_buf,    &karg.keys_addr));
  CHECK(vx_buffer_address(headers_buf, &karg.headers_addr));
  CHECK(vx_buffer_address(pids_buf,    &karg.pids_addr));
  CHECK(vx_buffer_address(meta_buf,    &karg.meta_addr));

  CHECK(vx_enqueue_write(q, prims_buf, 0, prims.data(), n * sizeof(binsort_prim_t), 0, nullptr, nullptr));

  // Single CTA (this increment): grid = 1, block = a valid device CTA size.
  uint32_t one = 1, grid[1], block[1];
  CHECK(vx_device_max_occupancy_grid(dev, 1, &one, grid, block));

  vx_launch_info_t li{};
  li.struct_size = sizeof(li);
  li.kernel      = kern;
  li.args_host   = &karg;
  li.args_size   = sizeof(karg);
  li.ndim        = 1;
  li.grid_dim[0] = 1;
  li.block_dim[0] = block[0];

  std::vector<uint32_t> h_meta(2, 0);
  std::vector<binsort_header_t> h_headers(max_bins);
  std::vector<uint32_t> h_pids(ref.P ? ref.P : 1, 0);

  vx_event_h lev = nullptr, ev_m = nullptr, ev_h = nullptr, ev_p = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &lev));
  CHECK(vx_enqueue_read(q, h_meta.data(),    meta_buf,    0, 2 * sizeof(uint32_t),                 1, &lev, &ev_m));
  CHECK(vx_enqueue_read(q, h_headers.data(), headers_buf, 0, ref.nbins * sizeof(binsort_header_t), 1, &lev, &ev_h));
  CHECK(vx_enqueue_read(q, h_pids.data(),    pids_buf,    0, ref.P * sizeof(uint32_t),             1, &lev, &ev_p));
  CHECK(vx_event_wait_value(ev_m, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_h, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_p, 1, VX_TIMEOUT_INFINITE));

  int errors = 0;
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

  vx_event_release(ev_p); vx_event_release(ev_h); vx_event_release(ev_m);
  vx_event_release(lev);
  vx_buffer_release(prims_buf); vx_buffer_release(count_buf); vx_buffer_release(offset_buf);
  vx_buffer_release(keys_buf); vx_buffer_release(headers_buf); vx_buffer_release(pids_buf); vx_buffer_release(meta_buf);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_release(dev);

  std::printf("RESULT: %s\n", errors == 0 ? "PASS" : "FAIL");
  return errors == 0 ? 0 : 1;
}
