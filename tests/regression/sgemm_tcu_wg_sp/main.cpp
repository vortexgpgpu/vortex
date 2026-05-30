#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <rvfloats.h>
#include <string.h>
#include <tensor.h>
#include <tensor_cfg.h>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex2.h>

#define FLOAT_ULP 10
#define MAX_ERRORS 100

#define RT_CHECK(_expr)                                       \
  do {                                                        \
    int _ret = _expr;                                         \
    if (0 == _ret)                                            \
      break;                                                  \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                                \
    exit(-1);                                                 \
  } while (false)

using namespace vortex;
namespace vt = tensor;

// WGMMA geometry
using wg_cfg_t = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, WGMMA_NRC>;

// Sparse parameters derived from wg_cfg_t
static constexpr uint32_t kRtlIRatio     = 32 / vt::ITYPE::bits;
static constexpr uint32_t kTcK           = wg_cfg_t::tcK;
static constexpr uint32_t kTcM           = wg_cfg_t::tcM;
static constexpr uint32_t kMSteps        = wg_cfg_t::m_steps;
static constexpr uint32_t kKSteps        = wg_cfg_t::k_steps;
static constexpr uint32_t kHalfKSteps    = kKSteps / 2;
static constexpr uint32_t kMetaRowBits   = kTcK * 2 * kRtlIRatio;
static constexpr uint32_t kMetaStrWords  = (kTcM * kMetaRowBits + 31) / 32;
static constexpr uint32_t kWgMetaBanks   = kMSteps * kHalfKSteps;
// Per-thread metadata layout: one TCU_LD slot covers NT 32-bit words,
// where the word at offset T is destined for SRAM cell
// (bank = T % PWD_wmma, col = T / PWD_wmma). PWD_wmma is the WMMA-cfg
// per-warp depth (= NT for the canonical configs WGMMA shares with WMMA).
static constexpr uint32_t kWordsPerTile  = VX_CFG_NUM_THREADS;

// WMMA-cfg geometry that the metadata SRAM is sized on — WGMMA's TCU_LD
// AGU + meta SRAM are dimensioned against the wmma_config_t<NT>, so the
// host pack must respect that bank layout regardless of WGMMA's own
// (smaller) m_steps/k_steps.
using kcfg = vt::wmma_config_t<VX_CFG_NUM_THREADS>;
static constexpr uint32_t kPwdWmma       = kcfg::m_steps * (kcfg::k_steps / 2);
static constexpr uint32_t kRtlHalfKWmma  = kcfg::k_steps / 2;

// dense elements covered by one sparse step
static constexpr uint32_t kDensePerSpStep = kTcK * kRtlIRatio * 2;

using itype_t = vt::ITYPE::dtype;
using otype_t = vt::OTYPE::dtype;

// CPU reference matmul using pruned (zero-padded) A
static void matmul_cpu(otype_t *C, const itype_t *A_pruned, const itype_t *B,
                       uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      otype_t sum = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        auto a = A_pruned[m * K + k];
        auto b = B[k * N + n];
        auto fa = bit_cast<float>(rv_htof_s(a, 0, nullptr));
        auto fb = bit_cast<float>(rv_htof_s(b, 0, nullptr));
        sum += fa * fb;
      }
      C[m * N + n] = sum;
    }
  }
}

// Pack sparse masks into the per-thread TCU_LD layout.
// For each (tile_row, k_tile): NT 32-bit words, where the word at offset
// T is loaded by AGU lane T and lands in SRAM cell
// (bank = T % PWD_wmma, col = T / PWD_wmma).
// WGMMA only uses SRAM banks of the form sm * RTL_HALF_K_wmma (sk_wg=0),
// so the remaining slots in the tile are zero.
static void pack_metadata_wg(std::vector<uint32_t> &h_meta,
                              const std::vector<uint8_t> &masks,
                              uint32_t M, uint32_t K) {
  constexpr uint32_t tileM      = wg_cfg_t::xtileM;
  constexpr uint32_t tileK_elem = wg_cfg_t::tileK;
  uint32_t num_tile_rows = M / tileM;
  uint32_t num_k_tiles   = K / tileK_elem;
  uint32_t num_groups_per_row = K / 4; // groups of 4 fp16 elements per row

  h_meta.assign(num_tile_rows * num_k_tiles * kWordsPerTile, 0);

  for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
      uint32_t tile_base = (tr * num_k_tiles + kt) * kWordsPerTile;

      for (uint32_t sm = 0; sm < kMSteps; ++sm) {
        for (uint32_t sk = 0; sk < kHalfKSteps; ++sk) {
          // SRAM bank for this (sm, sk_wg) in the WMMA-cfg-sized meta SRAM.
          uint32_t sram_bank = sm * kRtlHalfKWmma + sk;
          // dense element start for this (kt, sk)
          uint32_t k_elem_start = kt * tileK_elem + sk * kDensePerSpStep;
          uint32_t groups_in_step = kDensePerSpStep / 4;

          for (uint32_t i = 0; i < kTcM; ++i) {
            uint32_t physical_row = tr * tileM + sm * kTcM + i;
            uint32_t row_base = i * kMetaRowBits;

            for (uint32_t g = 0; g < groups_in_step; ++g) {
              uint32_t global_group = (k_elem_start / 4) + g;
              uint8_t mask = masks[physical_row * num_groups_per_row + global_group];

              for (int p = 0; p < 4; ++p) {
                if (mask & (1 << p)) {
                  uint32_t elt      = g * 4 + p;
                  uint32_t k_reg    = elt / (2 * kRtlIRatio);
                  uint32_t pos_in_k = elt % (2 * kRtlIRatio);
                  uint32_t meta_bit;
                  if (pos_in_k < kRtlIRatio) {
                    meta_bit = k_reg * kRtlIRatio + pos_in_k;
                  } else {
                    meta_bit = (kTcK + k_reg) * kRtlIRatio + (pos_in_k - kRtlIRatio);
                  }
                  uint32_t block_bit = row_base + meta_bit;
                  uint32_t word_idx  = block_bit / 32;
                  uint32_t bit_pos   = block_bit % 32;
                  // Per-thread layout: lane T = sram_bank + word_idx * PWD_wmma.
                  uint32_t lane = sram_bank + word_idx * kPwdWmma;
                  h_meta[tile_base + lane] |= (1u << bit_pos);
                }
              }
            }
          }
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

const char *kernel_file = "kernel.vxbin";

uint32_t xm = 64;
uint32_t xn = 64;
uint32_t xk = 64;
// `warps` (= warps per CTA = WGMMA group size) is derived at runtime from
// VX_CAPS_ISSUE_WIDTH after the device opens. WGMMA requires CTA-warp count
// to match the hardware's TCU BLOCK_SIZE (= ISSUE_WIDTH), so it's not a
// user-facing knob.
uint32_t warps = 0;

vx_device_h device      = nullptr;
vx_buffer_h A_buffer    = nullptr;
vx_buffer_h B_buffer    = nullptr;
vx_buffer_h C_buffer    = nullptr;
vx_buffer_h meta_buffer = nullptr;
vx_queue_h  queue       = nullptr;
vx_module_h module_     = nullptr;
vx_kernel_h kernel      = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex Sgemm2 TCU Sparse (WGMMA_SP) Test." << std::endl;
  std::cout << "Usage: [-m M] [-n N] [-k K] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:h")) != -1) {
    switch (c) {
    case 'm': xm = atoi(optarg); break;
    case 'n': xn = atoi(optarg); break;
    case 'k': xk = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (A_buffer)    vx_buffer_release(A_buffer);
    if (B_buffer)    vx_buffer_release(B_buffer);
    if (C_buffer)    vx_buffer_release(C_buffer);
    if (meta_buffer) vx_buffer_release(meta_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if (!(isa_flags & VX_ISA_EXT_TCU)) {
    std::cout << "TCU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t NT;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &NT));
  if (NT != VX_CFG_NUM_THREADS) {
    std::cout << "Error: device warp size (" << NT << ") must match VX_CFG_NUM_THREADS=" << VX_CFG_NUM_THREADS << "!" << std::endl;
    return -1;
  }

  uint64_t num_warps;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));

  // WGMMA group size = ISSUE_WIDTH. The hardware lockstep gate dispatches
  // BLOCK_SIZE = ISSUE_WIDTH warps in parallel per uop, so the CTA must
  // launch exactly that many active warps. Derived here, not user-tunable.
  uint64_t issue_width;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISSUE_WIDTH, &issue_width));
  warps = (uint32_t)issue_width;
  if (warps > num_warps) {
    std::cout << "Error: WGMMA group size (" << warps
              << " = VX_CFG_ISSUE_WIDTH) exceeds device's per-core warp count ("
              << num_warps << ")" << std::endl;
    return -1;
  }

  uint32_t M = xm, N = xn, K = xk;

  constexpr uint32_t tileM      = wg_cfg_t::xtileM;
  constexpr uint32_t tileN      = wg_cfg_t::xtileN;
  constexpr uint32_t tileK_elem = wg_cfg_t::tileK;

  uint32_t cta_M = warps * tileM;

  if ((M % cta_M) != 0) { std::cout << "M must be multiple of cta_M=" << cta_M << std::endl; return -1; }
  if ((N % tileN) != 0) { std::cout << "N must be multiple of " << tileN << std::endl; return -1; }
  if ((K % tileK_elem) != 0) { std::cout << "K must be multiple of " << tileK_elem << std::endl; return -1; }

  size_t sizeA_full = M * K;
  size_t sizeA_sp   = M * (K / 2);  // compressed
  size_t sizeB      = K * N;
  size_t sizeC      = M * N;

  uint32_t num_tile_rows = M / tileM;
  uint32_t num_k_tiles   = K / tileK_elem;
  size_t meta_words      = (size_t)num_tile_rows * num_k_tiles * kWordsPerTile;

  uint32_t grid_dim[2]  = {N / tileN, M / cta_M};
  uint32_t block_dim[2] = {warps * (uint32_t)NT, 1};

  std::cout << "ITYPE=" << vt::ITYPE::bits << "b, OTYPE=" << vt::OTYPE::bits << "b" << std::endl;
  std::cout << "tile M=" << tileM << " N=" << tileN << " K=" << tileK_elem << std::endl;
  std::cout << "cta_M=" << cta_M << " (warps=" << warps << ")" << std::endl;
  std::cout << "matrix A: " << M << "x" << K << " (compressed " << M << "x" << K/2 << ")" << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;
  std::cout << "wg_meta_banks=" << kWgMetaBanks << " meta_words_per_tile=" << kWordsPerTile << std::endl;

  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, sizeA_sp * sizeof(itype_t), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_buffer_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_buffer_create(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_buffer_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_buffer_create(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_buffer_address(C_buffer, &kernel_arg.C_addr));
  RT_CHECK(vx_buffer_create(device, meta_words * sizeof(uint32_t), VX_MEM_READ, &meta_buffer));
  RT_CHECK(vx_buffer_address(meta_buffer, &kernel_arg.meta_sp_addr));

  // generate dense A and B
  std::vector<itype_t> h_A_full(sizeA_full);
  std::vector<itype_t> h_B(sizeB);
  for (uint32_t i = 0; i < sizeA_full; ++i) {
    auto fv = float(rand()) / RAND_MAX;
    h_A_full[i] = rv_ftoh_s(bit_cast<uint32_t>(fv), 0, nullptr);
  }
  for (uint32_t i = 0; i < sizeB; ++i) {
    auto fv = float(rand()) / RAND_MAX;
    h_B[i] = rv_ftoh_s(bit_cast<uint32_t>(fv), 0, nullptr);
  }

  // prune A in-place (2:4) then compress
  if (!vt::prune_2to4_matrix<vt::ITYPE>(h_A_full.data(), M, K)) {
    std::cerr << "prune_2to4_matrix failed" << std::endl;
    cleanup(); return -1;
  }
  std::vector<itype_t> h_A_sp(sizeA_sp);
  std::vector<uint8_t> masks;
  if (!vt::compress_2to4_matrix<vt::ITYPE>(h_A_sp.data(), h_A_full.data(), masks, M, K)) {
    std::cerr << "compress_2to4_matrix failed" << std::endl;
    cleanup(); return -1;
  }

  // pack metadata into WGMMA smem bank layout
  std::vector<uint32_t> h_meta;
  pack_metadata_wg(h_meta, masks, M, K);

  // upload to device
  std::cout << "upload matrix A (compressed)" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, A_buffer, 0, h_A_sp.data(), sizeA_sp * sizeof(itype_t), 0, nullptr, nullptr));
  std::cout << "upload matrix B" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, B_buffer, 0, h_B.data(), sizeB * sizeof(itype_t), 0, nullptr, nullptr));
  std::cout << "upload metadata" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, meta_buffer, 0, h_meta.data(), meta_words * sizeof(uint32_t), 0, nullptr, nullptr));

  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // smem: per-warp [A_compressed][metadata] sections, then shared B
  uint32_t smem_a_bytes      = tileM * (tileK_elem / 2) * sizeof(itype_t);
  uint32_t smem_meta_bytes   = kWordsPerTile * 4;
  // SMEM bank-row = NUM_THREADS × LSU_WORD_SIZE (= XLEN/8). Must match the kernel
  // (kernel.cpp) — host-side lmem_size determines per-CTA allocation, which has to
  // include the same bank-row padding the kernel assumes.
  uint32_t smem_bank_bytes   = VX_CFG_NUM_THREADS * (VX_CFG_XLEN / 8);
  uint32_t per_warp_section  = ((smem_a_bytes + smem_meta_bytes + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;
  uint32_t smem_b_bytes      = tileK_elem * tileN * sizeof(itype_t);
  uint32_t smem_b_off        = ((warps * per_warp_section + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;
  uint32_t smem_size         = smem_b_off + smem_b_bytes;

  auto time_start = std::chrono::high_resolution_clock::now();

  // download result buffer — must outlive the async read enqueued below
  std::vector<otype_t> h_C(sizeC);

  std::cout << "launch kernel" << std::endl;
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 2;
    li.grid_dim[0]  = grid_dim[0];
    li.grid_dim[1]  = grid_dim[1];
    li.block_dim[0] = block_dim[0];
    li.block_dim[1] = block_dim[1];
    li.lmem_size    = smem_size;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  // download result — chained after the launch
  std::cout << "download destination buffer" << std::endl;
  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t), 1, &launch_ev, &read_ev));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // verify against CPU reference using pruned A (zeros at sparse positions)
  std::cout << "verify result" << std::endl;
  std::vector<otype_t> h_ref(sizeC);
  matmul_cpu(h_ref.data(), h_A_full.data(), h_B.data(), M, N, K);

  int errors = 0;
  for (uint32_t i = 0; i < sizeC; ++i) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = h_C[i];
    fb.f = h_ref[i];
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%u] expected=%f, actual=%f\n", i, h_ref[i], h_C[i]);
      }
      ++errors;
    }
  }

  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " / " << sizeC << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
