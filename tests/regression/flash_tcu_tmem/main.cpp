// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// flash attention UMMA regression test.

#include <vortex2.h>
#include <tensor_cfg.h>
#include <rvfloats.h>
#include "common.h"
#include "fpcvt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

namespace vt = vortex::tensor;

using umma_cfg  = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, UMMA_NRC>;
using input_t = typename vt::ITYPE::dtype;

// The CTA dispatcher rounds each CTA's LMEM footprint up to a MEM_BLOCK_SIZE
// multiple to get the slot pitch.
static constexpr uint32_t kBlockAlign = VX_CFG_MEM_BLOCK_SIZE;

// Tile quanta implied by the UMMA geometry.
static constexpr uint32_t kQuantR = umma_cfg::xtileM;
static constexpr uint32_t kQuantC = (umma_cfg::xtileN > umma_cfg::tileK) ? umma_cfg::xtileN : umma_cfg::tileK;

#define CHECK(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n", \
                     __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        std::exit(1); \
    } \
} while (0)

namespace {
const char* kernel_file = "kernel.vxbin";
uint32_t N = 64;
uint32_t d = 32;
uint32_t r_override = 0, c_override = 0, d_override = 0;

void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "N:D:k:r:c:d:h")) != -1) {
    switch (c) {
      case 'N': N           = std::atoi(optarg); break;
      case 'D': d           = std::atoi(optarg); break;
      case 'k': kernel_file = optarg;            break;
      case 'r': r_override  = std::atoi(optarg); break;
      case 'c': c_override  = std::atoi(optarg); break;
      case 'd': d_override  = std::atoi(optarg); break;
      default:
        std::cout << "Usage: [-k: kernel] [-N: sequence_length] [-D: head_dim] "
                     "[-r: block_size_r] [-c: block_size_c] [-d: head_dim_tile] [-h]" << std::endl;
        std::exit(c == 'h' ? 0 : -1);
    }
  }
}

// ITYPE <-> fp32 on host
input_t pack_h(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  if      constexpr (std::is_same_v<vt::ITYPE, vt::tf32>) return (input_t)rv_ftotf32_s(u, 0, nullptr);
  else if constexpr (std::is_same_v<vt::ITYPE, vt::fp16>) return (input_t)rv_ftoh_s(u, 0, nullptr);
  else if constexpr (std::is_same_v<vt::ITYPE, vt::bf16>) return (input_t)rv_ftob_s(u, 0, nullptr);
  else { std::cerr << "unsupported ITYPE\n"; std::exit(1); }
}
float unpack_h(input_t v) {
  uint32_t u;
  if      constexpr (std::is_same_v<vt::ITYPE, vt::tf32>) u = rv_tf32tof_s((uint32_t)v, 0, nullptr);
  else if constexpr (std::is_same_v<vt::ITYPE, vt::fp16>) u = rv_htof_s((uint16_t)v, 0, nullptr);
  else if constexpr (std::is_same_v<vt::ITYPE, vt::bf16>) u = rv_btof_s((uint16_t)v, 0, nullptr);
  else { std::cerr << "unsupported ITYPE\n"; std::exit(1); }
  float f; std::memcpy(&f, &u, 4); return f;
}

// fpcvt.h must bit-match the host softfloat, or a device-packed P diverges from
// the reference.
bool selftest_fpcvt() {
  auto mine = [](float f) -> uint32_t {
    if      constexpr (std::is_same_v<vt::ITYPE, vt::tf32>) return fpcvt::f32_to_tf32(f);
    else if constexpr (std::is_same_v<vt::ITYPE, vt::fp16>) return fpcvt::f32_to_fp16(f);
    else                                                    return fpcvt::f32_to_bf16(f);
  };
  long bad = 0, n = 0;
  auto chk = [&](float f) {
    ++n;
    if (mine(f) != (uint32_t)pack_h(f)) {
      if (++bad <= 4) {
        uint32_t u; std::memcpy(&u, &f, 4);
        std::printf("fpcvt mismatch: f=%.9g (0x%08x) device=0x%05x host=0x%05x\n",
                    (double)f, u, mine(f), (uint32_t)pack_h(f));
      }
    }
  };
  std::srand(7);
  for (int i = 0; i < 20000; ++i) chk((float)std::rand() / RAND_MAX);
  for (int e = 0; e < 256; ++e)
    for (uint32_t m : {0u, 1u, 0x1000u, 0x0FFFu, 0x400000u, 0x7FFFFFu, 0x2AAAAAu})
      for (uint32_t s : {0u, 1u}) {
        uint32_t u = (s << 31) | ((uint32_t)e << 23) | m; float f; std::memcpy(&f, &u, 4); chk(f);
      }
  for (int e = -140; e <= 130; ++e) { chk(std::ldexp(1.0f, e)); chk(-std::ldexp(1.0f, e)); }
  std::cout << "fpcvt self-test (" << vt::ITYPE::name << "): " << n << " cases, "
            << bad << " mismatched" << std::endl;
  return bad == 0;
}

void flash_attention_ref(float* out, const input_t* Q, const input_t* K, const input_t* V,
                         uint32_t N, uint32_t d) {
  std::vector<float> scores(N), probs(N);
  for (uint32_t i = 0; i < N; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float sum = 0.f;
      for (uint32_t k = 0; k < d; ++k)
        sum += unpack_h(Q[i * d + k]) * unpack_h(K[j * d + k]);
      scores[j] = sum;
    }
    float mx = scores[0];
    for (uint32_t j = 1; j < N; ++j) mx = std::max(mx, scores[j]);
    float esum = 0.f;
    for (uint32_t j = 0; j < N; ++j) { probs[j] = std::exp(scores[j] - mx); esum += probs[j]; }
    for (uint32_t k = 0; k < d; ++k) {
      float sum = 0.f;
      for (uint32_t j = 0; j < N; ++j)
        sum += unpack_h(pack_h(probs[j])) * unpack_h(V[j * d + k]);
      out[i * d + k] = sum / esum;
    }
  }
}
} // namespace

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  std::cout << "flash_tcu_tmem: " << N << "x" << d
            << " ITYPE=" << vt::ITYPE::name << " OTYPE=" << vt::OTYPE::name
            << " NRC=" << UMMA_NRC << std::endl;
  std::cout << "UMMA geometry: tcM=" << umma_cfg::tcM << " tcN=" << umma_cfg::tcN
            << " xtileM=" << umma_cfg::xtileM << " xtileN=" << umma_cfg::xtileN
            << " tileK=" << umma_cfg::tileK
            << "  -> block_size_r step " << kQuantR
            << ", block_size_c/head_dim_tile step " << kQuantC << std::endl;

  if (!selftest_fpcvt()) { std::cout << "FAILED (fpcvt)" << std::endl; return 1; }

  if (N == 0 || d == 0) { printf("Error: N and D must be nonzero\n"); return -1; }

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  auto t_start = std::chrono::high_resolution_clock::now();

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  uint64_t isa_flags = 0;
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_TCU) == 0) {
    std::cout << "TCU extension not supported!" << std::endl; return -1;
  }

  uint64_t nt_q = 0, nw_q = 0, lmem_size = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,    &nt_q));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,      &nw_q));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_size));
  uint32_t num_threads = (uint32_t)nt_q, num_warps = (uint32_t)nw_q;
  if (num_threads != VX_CFG_NUM_THREADS) {
    std::cout << "Error: device threads (" << num_threads << ") != VX_CFG_NUM_THREADS="
              << VX_CFG_NUM_THREADS << std::endl; return -1;
  }

  uint64_t iw_q = 0, tmem_cols = 0;
  CHECK(vx_device_query(dev, VX_CAPS_ISSUE_WIDTH,   &iw_q));
  CHECK(vx_device_query(dev, VX_CAPS_TCU_TMEM_COLS, &tmem_cols));
  if (tmem_cols == 0) {
    std::cout << "Error: device reports 0 TMEM columns (VX_CFG_TCU_TMEM_ENABLE is off)"
              << std::endl; return -1;
  }

  // Warps per CTA must equal ISSUE_WIDTH
  uint32_t cta_warps = (uint32_t)iw_q;
  if (cta_warps > num_warps) {
    std::cout << "Error: ISSUE_WIDTH (" << cta_warps << ") exceeds the core's warp count ("
              << num_warps << "), so a full warpgroup cannot be resident" << std::endl;
    return -1;
  }
  const uint32_t req_Br = cta_warps * kQuantR;

  // Require clean divisibility
  auto round_up = [](uint32_t v, uint32_t m) { return ((v + m - 1) / m) * m; };
  if (d % kQuantC) {
    uint32_t d2 = round_up(d, kQuantC);
    std::cout << "Note: head_dim " << d << " -> " << d2 << " (multiple of " << kQuantC << ")" << std::endl;
    d = d2;
  }

  // block_size_r steps by xtileM and block_size_c/head_dim_tile are rounded to
  // UMMA quantum.
  auto lmem_for = [&](uint32_t r, uint32_t c, uint32_t dtile) -> uint64_t {
    return (uint64_t)sizeof(input_t)  * ((uint64_t)r * d + 2ull * c * dtile + (uint64_t)r * c)
         + (uint64_t)sizeof(float)    * ((uint64_t)r * c + (uint64_t)r * d + 3ull * r);
  };

  uint32_t Br = 0, Bc = 0, dt = 0, occupancy = 0;
  uint64_t local_mem = 0;
  {
    // block_size_r is pinned to one full warpgroup
    if (r_override && r_override != req_Br) {
      std::cout << "Error: block_size_r must be " << req_Br << " (ISSUE_WIDTH="
                << cta_warps << " warps x xtileM=" << kQuantR
                << "); a CTA narrower or wider than the warpgroup is not a valid"
                   " UMMA configuration" << std::endl;
      return -1;
    }
    const uint32_t r = req_Br;
    if (N % r) {
      std::cout << "Error: seq_len " << N << " is not a multiple of block_size_r "
                << r << std::endl;
      return -1;
    }
    uint32_t c = c_override ? c_override : std::max(kQuantC, round_up(num_threads, kQuantC));
    if ((c % kQuantC) == 0 && (N % c) == 0) {
      for (uint32_t cand = d_override ? d_override : d; cand >= kQuantC; cand -= kQuantC) {
        if (d % cand) continue;
        uint64_t use = lmem_for(r, c, cand);
        if (use > lmem_size) { if (d_override) break; else continue; }
        Br = r; Bc = c; dt = cand; local_mem = use;
        occupancy = std::min((uint32_t)std::max<uint64_t>(1, lmem_size / use), num_warps / cta_warps);
        break;
      }
    }
  }
  if (!dt) {
    printf("Error: with block_size_r pinned to %u (%u warps x xtileM=%u), no "
           "(block_size_c, head_dim_tile) on the UMMA quantum of %u fits the LMEM "
           "budget (%llu bytes) for N=%u D=%u\n",
           req_Br, cta_warps, kQuantR, kQuantC,
           (unsigned long long)lmem_size, N, d);
    return -1;
  }

  // Each concurrently-resident CTA holds its own xtileN-column allocation
  if ((uint64_t)occupancy * umma_cfg::xtileN > tmem_cols) {
    std::cout << "Error: " << occupancy << " concurrent CTAs x xtileN="
              << umma_cfg::xtileN << " = " << (uint64_t)occupancy * umma_cfg::xtileN
              << " TMEM columns, which exceeds the device's " << tmem_cols
              << ". Reduce UMMA_NRC or VX_CFG_NUM_WARPS." << std::endl;
    return -1;
  }

  std::cout << "num_threads=" << num_threads << " num_warps=" << num_warps
            << " lmem_size=" << lmem_size << " bytes" << std::endl;
  std::cout << "tmem_cols=" << tmem_cols << " (using " << umma_cfg::xtileN
            << " per CTA) warps/CTA=" << cta_warps
            << " (== issue_width)" << std::endl;
  std::cout << "block_size_r=" << Br << " (" << (Br / kQuantR) << " warps)"
            << " block_size_c=" << Bc << " head_dim_tile=" << dt << std::endl;
  std::cout << "local memory: " << local_mem << " bytes, occupancy=" << occupancy << std::endl;

  uint32_t size = N * d;
  vx_buffer_h Qb = nullptr, Kb = nullptr, Vb = nullptr, Ob = nullptr;
  CHECK(vx_buffer_create(dev, size * sizeof(input_t), VX_MEM_READ,  &Qb));
  CHECK(vx_buffer_create(dev, size * sizeof(input_t), VX_MEM_READ,  &Kb));
  CHECK(vx_buffer_create(dev, size * sizeof(input_t), VX_MEM_READ,  &Vb));
  CHECK(vx_buffer_create(dev, size * sizeof(float),   VX_MEM_WRITE, &Ob));

  vx_module_h mod = nullptr; vx_kernel_h kernel = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kernel));

  kernel_arg_t ka{};
  ka.seq_len = N; ka.head_dim = d; ka.head_dim_tile = dt;
  ka.block_size_r = Br; ka.block_size_c = Bc;
  CHECK(vx_buffer_address(Qb, &ka.Q_addr));
  CHECK(vx_buffer_address(Kb, &ka.K_addr));
  CHECK(vx_buffer_address(Vb, &ka.V_addr));
  CHECK(vx_buffer_address(Ob, &ka.O_addr));

  std::vector<input_t> hQ(size), hK(size), hV(size);
  std::vector<float>   hO(size), hRef(size);
  for (uint32_t i = 0; i < size; ++i) {
    hQ[i] = pack_h((float)std::rand() / RAND_MAX);
    hK[i] = pack_h((float)std::rand() / RAND_MAX);
    hV[i] = pack_h((float)std::rand() / RAND_MAX);
  }

  CHECK(vx_enqueue_write(q, Qb, 0, hQ.data(), size * sizeof(input_t), 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, Kb, 0, hK.data(), size * sizeof(input_t), 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, Vb, 0, hV.data(), size * sizeof(input_t), 0, nullptr, nullptr));

  vx_launch_info_t li{};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &ka;
  li.args_size    = sizeof(ka);
  li.ndim         = 2;
  li.grid_dim[0]  = N / Br;
  li.grid_dim[1]  = 1;
  li.block_dim[0] = num_threads;
  li.block_dim[1] = Br / kQuantR;
  li.lmem_size    = local_mem;

  vx_event_h lev = nullptr, rev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &lev));
  CHECK(vx_enqueue_read(q, hO.data(), Ob, 0, size * sizeof(float), 1, &lev, &rev));
  CHECK(vx_event_wait_value(rev, 1, VX_TIMEOUT_INFINITE));
  auto t_end = std::chrono::high_resolution_clock::now();
  std::printf("Elapsed: %ld ms\n",
      (long)std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
  vx_event_release(rev); vx_event_release(lev);

  flash_attention_ref(hRef.data(), hQ.data(), hK.data(), hV.data(), N, d);

  // Inputs are ITYPE-rounded in both paths, so residual is FEDP summation
  // order over d and over N. Scale tolerance with the reduction length.
  const float tol = 1e-3f * std::sqrt((float)std::max(N, d));
  int errors = 0; float worst = 0.f;
  for (uint32_t i = 0; i < size; ++i) {
    float e = std::fabs(hRef[i] - hO[i]) / std::max(1e-6f, std::fabs(hRef[i]));
    worst = std::max(worst, e);
    if (e > tol) {
      if (errors < 16)
        std::printf("*** error: [%u] expected=%f, actual=%f (rel %.3g)\n", i, hRef[i], hO[i], e);
      ++errors;
    }
  }
  std::printf("worst relative error: %.3g (tolerance %.3g)\n", worst, tol);

  vx_buffer_release(Qb); vx_buffer_release(Kb); vx_buffer_release(Vb); vx_buffer_release(Ob);
  vx_kernel_release(kernel); vx_module_release(mod); vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);

  if (errors) { std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl; return errors; }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
