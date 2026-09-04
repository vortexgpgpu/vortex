#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <VX_types.h>
#include <vortex2.h>
#include <dxa.h>

#include "common.h"

#define RT_CHECK(expr) do {                                            \
  int ret = (expr);                                                    \
  if (ret == 0) break;                                                 \
  std::cerr << "Error: '" << #expr << "' returned " << ret << "\n";  \
  cleanup();                                                           \
  return -1;                                                           \
} while (false)

static vx_device_h device = nullptr;
static vx_queue_h queue = nullptr;
static vx_module_h module = nullptr;
static vx_kernel_h kernel = nullptr;
static vx_buffer_h out_buf = nullptr;
static vx_buffer_h scale_buf = nullptr;
static vx_buffer_h state0_buf = nullptr;
static vx_buffer_h state1_buf = nullptr;
static vx_buffer_h meta_buf = nullptr;

static void cleanup() {
  if (out_buf) vx_buffer_release(out_buf);
  if (scale_buf) vx_buffer_release(scale_buf);
  if (state0_buf) vx_buffer_release(state0_buf);
  if (state1_buf) vx_buffer_release(state1_buf);
  if (meta_buf) vx_buffer_release(meta_buf);
  if (kernel) vx_kernel_release(kernel);
  if (module) vx_module_release(module);
  if (queue) vx_queue_release(queue);
  if (device) vx_device_release(device);
  out_buf = scale_buf = state0_buf = state1_buf = meta_buf = nullptr;
  kernel = nullptr;
  module = nullptr;
  queue = nullptr;
  device = nullptr;
}

static uint32_t out_value(uint32_t iter, uint32_t word) {
  return 0x10000000u + iter * 0x10000u + word;
}
static uint32_t scale_value(uint32_t iter, uint32_t word) {
  return 0x20000000u + iter * 0x10000u + word;
}
static uint32_t state0_value(uint32_t iter, uint32_t word) {
  return 0x30000000u + iter * 0x10000u + word;
}
static uint32_t state1_value(uint32_t iter, uint32_t word) {
  return 0x40000000u + iter * 0x10000u + word;
}
static uint32_t meta_value(uint32_t iter, uint32_t word) {
  return 0x50000000u + iter * 0x10000u + word;
}

template <typename T>
static int make_stream(vx_buffer_h* buf, uint32_t elements,
                       uint64_t* address, std::vector<T>* initial) {
  constexpr uint32_t kCanary = 0xdeadbeefu;
  const uint32_t total = elements + 2;
  initial->assign(total, static_cast<T>(kCanary));
  int ret = vx_buffer_create(device, total * sizeof(T),
                             VX_MEM_READ_WRITE | VX_MEM_PHYS, buf);
  if (ret) return ret;
  ret = vx_buffer_address(*buf, address);
  if (ret) return ret;
  ret = vx_enqueue_write(queue, *buf, 0, initial->data(),
                         total * sizeof(T), 0, nullptr, nullptr);
  if (ret) return ret;
  *address += sizeof(T);
  return 0;
}

static int read_stream(vx_buffer_h buf, uint32_t elements,
                       std::vector<uint32_t>* values) {
  values->assign(elements + 2, 0);
  vx_event_h event = nullptr;
  int ret = vx_enqueue_read(queue, values->data(), buf, 0,
                            (elements + 2) * sizeof(uint32_t),
                            0, nullptr, &event);
  if (ret) return ret;
  ret = vx_event_wait_value(event, 1, VX_TIMEOUT_INFINITE);
  vx_event_release(event);
  return ret;
}

static int reset_stream(vx_buffer_h buf, uint32_t elements) {
  // Keep the reset in the same queue as the following launch. This makes the
  // two modes independent without racing the previous asynchronous readback.
  constexpr uint32_t kCanary = 0xdeadbeefu;
  std::vector<uint32_t> canary(elements + 2, kCanary);
  int ret = vx_enqueue_write(queue, buf, 0, canary.data(),
                             canary.size() * sizeof(uint32_t),
                             0, nullptr, nullptr);
  // The enqueue API borrows host memory until the command is consumed.  The
  // reset payload is stack-owned, so finish this setup command before the
  // vector goes out of scope; kernel execution remains fully asynchronous.
  if (ret == 0)
    ret = vx_queue_finish(queue, VX_TIMEOUT_INFINITE);
  return ret;
}

static int run_mode(uint32_t mode, uint64_t out_addr, uint64_t scale_addr,
                    uint64_t state0_addr, uint64_t state1_addr,
                    uint64_t meta_addr, uint32_t block_threads) {
  // Descriptor coordinates are element offsets. Leave one stride-sized
  // segment between iterations so an overrun cannot hide in the next result.
  int ret = 0;
  ret |= vortex::dxa::program_1d(device, 0, out_addr,
      S2G_WS_ITERS * S2G_WS_OUT_STRIDE, S2G_WS_OUT_WORDS, 4);
  ret |= vortex::dxa::program_1d(device, 1, scale_addr,
      S2G_WS_ITERS * S2G_WS_SCALE_STRIDE, S2G_WS_SCALE_WORDS, 4);
  ret |= vortex::dxa::program_1d(device, 2, state0_addr,
      S2G_WS_ITERS * S2G_WS_STATE0_STRIDE, S2G_WS_STATE0_WORDS, 4);
  ret |= vortex::dxa::program_1d(device, 3, state1_addr,
      S2G_WS_ITERS * S2G_WS_STATE1_STRIDE, S2G_WS_STATE1_WORDS, 4);
  ret |= vortex::dxa::program_1d(device, 4, meta_addr,
      S2G_WS_ITERS * S2G_WS_META_STRIDE, S2G_WS_META_WORDS, 4);
  if (ret) return ret;

  kernel_arg_t args{mode, S2G_WS_ITERS};
  vx_launch_info_t li{};
  li.struct_size = sizeof(li);
  li.kernel = kernel;
  li.args_host = &args;
  li.args_size = sizeof(args);
  li.ndim = 1;
  li.grid_dim[0] = 1;
  li.block_dim[0] = block_threads;
  li.lmem_size = (2 * (S2G_WS_OUT_WORDS + S2G_WS_SCALE_WORDS)
                 + 3 * (S2G_WS_STATE0_WORDS + S2G_WS_STATE1_WORDS
                        + S2G_WS_META_WORDS)) * sizeof(uint32_t);

  vx_event_h launch = nullptr;
  ret = vx_enqueue_launch(queue, &li, 0, nullptr, &launch);
  if (ret) return ret;
  ret = vx_event_wait_value(launch, 1, VX_TIMEOUT_INFINITE);
  vx_event_release(launch);
  return ret;
}

int main(int argc, char** argv) {
  const char* kernel_file = argc > 1 ? argv[1] : "kernel.vxbin";
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_threads = 0, num_warps = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  if (num_warps < S2G_WS_WARPS) {
    std::cout << "SKIPPED: dxa_s2g_bulk_groups_ws needs "
              << S2G_WS_WARPS << " warps\n";
    cleanup();
    return 0;
  }
  const uint32_t block_threads =
      static_cast<uint32_t>(num_threads * S2G_WS_WARPS);

  std::vector<uint32_t> init;
  uint64_t out_addr = 0, scale_addr = 0, state0_addr = 0;
  uint64_t state1_addr = 0, meta_addr = 0;
  const uint32_t out_elems = S2G_WS_ITERS * S2G_WS_OUT_STRIDE;
  const uint32_t scale_elems = S2G_WS_ITERS * S2G_WS_SCALE_STRIDE;
  const uint32_t state0_elems = S2G_WS_ITERS * S2G_WS_STATE0_STRIDE;
  const uint32_t state1_elems = S2G_WS_ITERS * S2G_WS_STATE1_STRIDE;
  const uint32_t meta_elems = S2G_WS_ITERS * S2G_WS_META_STRIDE;
  RT_CHECK(make_stream(&out_buf, out_elems, &out_addr, &init));
  RT_CHECK(make_stream(&scale_buf, scale_elems, &scale_addr, &init));
  RT_CHECK(make_stream(&state0_buf, state0_elems, &state0_addr, &init));
  RT_CHECK(make_stream(&state1_buf, state1_elems, &state1_addr, &init));
  RT_CHECK(make_stream(&meta_buf, meta_elems, &meta_addr, &init));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module));
  RT_CHECK(vx_module_get_kernel(module, "main", &kernel));
  RT_CHECK(vx_check_occupancy(device, block_threads,
      (2 * (S2G_WS_OUT_WORDS + S2G_WS_SCALE_WORDS)
       + 3 * (S2G_WS_STATE0_WORDS + S2G_WS_STATE1_WORDS
              + S2G_WS_META_WORDS)) * sizeof(uint32_t)));

  for (uint32_t mode = 0; mode < 2; ++mode) {
    RT_CHECK(reset_stream(out_buf, out_elems));
    RT_CHECK(reset_stream(scale_buf, scale_elems));
    RT_CHECK(reset_stream(state0_buf, state0_elems));
    RT_CHECK(reset_stream(state1_buf, state1_elems));
    RT_CHECK(reset_stream(meta_buf, meta_elems));
    RT_CHECK(run_mode(mode, out_addr, scale_addr, state0_addr,
                      state1_addr, meta_addr, block_threads));

    std::vector<uint32_t> out, scale, state0, state1, meta;
    RT_CHECK(read_stream(out_buf, out_elems, &out));
    RT_CHECK(read_stream(scale_buf, scale_elems, &scale));
    RT_CHECK(read_stream(state0_buf, state0_elems, &state0));
    RT_CHECK(read_stream(state1_buf, state1_elems, &state1));
    RT_CHECK(read_stream(meta_buf, meta_elems, &meta));

    uint32_t errors = 0;
    auto check_canaries = [&](const std::vector<uint32_t>& v,
                              uint32_t elems, const char* name) {
      if (v.front() != 0xdeadbeefu || v[elems + 1] != 0xdeadbeefu) {
        std::cerr << "canary corrupted in " << name << " first=0x"
                  << std::hex << v.front() << " last=0x" << v[elems + 1]
                  << std::dec << "\n";
        ++errors;
      }
    };
    check_canaries(out, out_elems, "output");
    check_canaries(scale, scale_elems, "scale");
    check_canaries(state0, state0_elems, "state0");
    check_canaries(state1, state1_elems, "state1");
    check_canaries(meta, meta_elems, "metadata");
    if (errors) {
      std::cerr << "sample output:";
      for (uint32_t i = 0; i < 12 && i < out.size(); ++i)
        std::cerr << " [" << i << "]=0x" << std::hex << out[i];
      std::cerr << std::dec << "\n";
    }
    for (uint32_t iter = 0; iter < S2G_WS_ITERS; ++iter) {
      const uint32_t ob = 1 + iter * S2G_WS_OUT_STRIDE;
      const uint32_t sb = 1 + iter * S2G_WS_SCALE_STRIDE;
      const uint32_t s0b = 1 + iter * S2G_WS_STATE0_STRIDE;
      const uint32_t s1b = 1 + iter * S2G_WS_STATE1_STRIDE;
      const uint32_t mb = 1 + iter * S2G_WS_META_STRIDE;
      for (uint32_t i = 0; i < S2G_WS_OUT_WORDS; ++i)
        errors += out[ob + i] != out_value(iter, i);
      for (uint32_t i = 0; i < S2G_WS_SCALE_WORDS; ++i)
        errors += scale[sb + i] != scale_value(iter, i);
      for (uint32_t i = 0; i < S2G_WS_STATE0_WORDS; ++i)
        errors += state0[s0b + i] != state0_value(iter, i);
      if ((iter & 1u) == 0) {
        for (uint32_t i = 0; i < S2G_WS_STATE1_WORDS; ++i)
          errors += state1[s1b + i] != state1_value(iter, i);
      } else {
        for (uint32_t i = 0; i < S2G_WS_STATE1_WORDS; ++i)
          errors += state1[s1b + i] != 0xdeadbeefu;
      }
      if ((iter & 3u) == 0) {
        for (uint32_t i = 0; i < S2G_WS_META_WORDS; ++i)
          errors += meta[mb + i] != meta_value(iter, i);
      } else {
        for (uint32_t i = 0; i < S2G_WS_META_WORDS; ++i)
          errors += meta[mb + i] != 0xdeadbeefu;
      }
    }
    if (errors) {
      std::cerr << "FAILED mode=" << mode << " errors=" << errors << "\n";
      cleanup();
      return -1;
    }
    std::cout << "mode=" << (mode ? "PIPELINED" : "SERIAL")
              << " PASS (two issuers; output groups=2 ops, state groups=1/2/3 ops)\n";
  }
  std::cout << "PASSED: dxa_s2g_bulk_groups_ws\n";
  cleanup();
  return 0;
}
