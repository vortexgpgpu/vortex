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
#include <vortex.h>

#define FLOAT_ULP 6
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

// WGMMA geometry: same template as the kernel context (NR=32)
using wg_cfg_t = vt::wmma_config_t<NUM_THREADS, vt::fp32, vt::fp32, 4, 32>;

// Sparse parameters based on runtime format (fp16)
static constexpr uint32_t kRtlIRatio     = 32 / vt::fp16::bits;           // 2
static constexpr uint32_t kTcK           = wg_cfg_t::tcK;                   // 2
static constexpr uint32_t kTcM           = wg_cfg_t::tcM;                   // 4
static constexpr uint32_t kMSteps        = wg_cfg_t::m_steps;               // 4
static constexpr uint32_t kKSteps        = wg_cfg_t::k_steps;               // 8
static constexpr uint32_t kHalfKSteps    = kKSteps / 2;                     // 4
static constexpr uint32_t kMetaRowBits   = kTcK * 2 * kRtlIRatio;          // 8
static constexpr uint32_t kMetaStrWords  = (kTcM * kMetaRowBits + 31) / 32; // 1
static constexpr uint32_t kWgMetaBanks   = kMSteps * kHalfKSteps;           // 16
static constexpr uint32_t kWordsPerTile  = kWgMetaBanks * kMetaStrWords;    // 16

// dense elements covered by one sparse step
static constexpr uint32_t kDensePerSpStep = kTcK * kRtlIRatio * 2; // 8

using itype_t = vt::fp16::dtype;   // uint16_t
using otype_t = vt::fp32::dtype;   // float

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

// Pack sparse masks into WGMMA smem bank layout.
// For each (tile_row, k_tile): kWordsPerTile words, organized as
//   [bank0: kMetaStrWords words][bank1: ...]...[bank(kWgMetaBanks-1)]
// bank = step_m * kHalfKSteps + step_k
// Within a bank: bit block_bit = row_i * kMetaRowBits + meta_bit
static void pack_metadata_wg(std::vector<uint32_t> &h_meta,
                              const std::vector<uint8_t> &masks,
                              uint32_t M, uint32_t K) {
  // Retrieve tileM/tileK from wg context (fp16 i_ratio)
  constexpr uint32_t tileM      = wg_cfg_t::xtileM;         // 16
  constexpr uint32_t tileK_elem = wg_cfg_t::tcK * kRtlIRatio * kHalfKSteps * 2; // = xtileK * i_ratio
  uint32_t num_tile_rows = M / tileM;
  uint32_t num_k_tiles   = K / tileK_elem;
  uint32_t num_groups_per_row = K / 4; // groups of 4 fp16 elements per row

  h_meta.assign(num_tile_rows * num_k_tiles * kWordsPerTile, 0);

  for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
      uint32_t tile_base = (tr * num_k_tiles + kt) * kWordsPerTile;

      for (uint32_t sm = 0; sm < kMSteps; ++sm) {
        for (uint32_t sk = 0; sk < kHalfKSteps; ++sk) {
          uint32_t bank = sm * kHalfKSteps + sk;
          uint32_t bank_word_base = tile_base + bank * kMetaStrWords;
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
                  h_meta[bank_word_base + word_idx] |= (1u << bit_pos);
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

uint32_t xm = 32;
uint32_t xn = 32;
uint32_t xk = 32;

vx_device_h device      = nullptr;
vx_buffer_h A_buffer    = nullptr;
vx_buffer_h B_buffer    = nullptr;
vx_buffer_h C_buffer    = nullptr;
vx_buffer_h meta_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
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
    vx_mem_free(A_buffer);
    vx_mem_free(B_buffer);
    vx_mem_free(C_buffer);
    vx_mem_free(meta_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if (!(isa_flags & VX_ISA_EXT_TCU)) {
    std::cout << "TCU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t NT;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT));
  if (NT != NUM_THREADS) {
    std::cout << "Error: device warp size (" << NT << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
    return -1;
  }

  uint32_t M = xm, N = xn, K = xk;

  // Tile dimension constants (fp16 context)
  constexpr uint32_t tileM     = wg_cfg_t::xtileM;
  constexpr uint32_t tileN     = wg_cfg_t::xtileN;
  constexpr uint32_t tileK_elem = kTcK * kRtlIRatio * kHalfKSteps * 2; // 32 fp16 elements

  if ((M % tileM) != 0) { std::cout << "M must be multiple of " << tileM << std::endl; return -1; }
  if ((N % tileN) != 0) { std::cout << "N must be multiple of " << tileN << std::endl; return -1; }
  if ((K % tileK_elem) != 0) { std::cout << "K must be multiple of " << tileK_elem << std::endl; return -1; }

  size_t sizeA_full = M * K;
  size_t sizeA_sp   = M * (K / 2);  // compressed
  size_t sizeB      = K * N;
  size_t sizeC      = M * N;

  uint32_t num_tile_rows = M / tileM;
  uint32_t num_k_tiles   = K / tileK_elem;
  size_t meta_words      = (size_t)num_tile_rows * num_k_tiles * kWordsPerTile;

  uint32_t grid_dim[2]  = {N / tileN, M / tileM};
  uint32_t block_dim[2] = {(uint32_t)NT, 1};

  std::cout << "ITYPE=fp16, OTYPE=fp32" << std::endl;
  std::cout << "tile M=" << tileM << " N=" << tileN << " K=" << tileK_elem << std::endl;
  std::cout << "matrix A: " << M << "x" << K << " (compressed " << M << "x" << K/2 << ")" << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;
  std::cout << "wg_meta_banks=" << kWgMetaBanks << " meta_words_per_tile=" << kWordsPerTile << std::endl;

  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, sizeA_sp * sizeof(itype_t), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));
  RT_CHECK(vx_mem_alloc(device, meta_words * sizeof(uint32_t), VX_MEM_READ, &meta_buffer));
  RT_CHECK(vx_mem_address(meta_buffer, &kernel_arg.meta_addr));

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
  if (!vt::prune_2to4_matrix<vt::fp16>(h_A_full.data(), M, K)) {
    std::cerr << "prune_2to4_matrix failed" << std::endl;
    cleanup(); return -1;
  }
  std::vector<itype_t> h_A_sp(sizeA_sp);
  std::vector<uint8_t> masks;
  if (!vt::compress_2to4_matrix<vt::fp16>(h_A_sp.data(), h_A_full.data(), masks, M, K)) {
    std::cerr << "compress_2to4_matrix failed" << std::endl;
    cleanup(); return -1;
  }

  // pack metadata into WGMMA smem bank layout
  std::vector<uint32_t> h_meta;
  pack_metadata_wg(h_meta, masks, M, K);

  // upload to device
  std::cout << "upload matrix A (compressed)" << std::endl;
  RT_CHECK(vx_copy_to_dev(A_buffer, h_A_sp.data(), 0, sizeA_sp * sizeof(itype_t)));
  std::cout << "upload matrix B" << std::endl;
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, sizeB * sizeof(itype_t)));
  std::cout << "upload metadata" << std::endl;
  RT_CHECK(vx_copy_to_dev(meta_buffer, h_meta.data(), 0, meta_words * sizeof(uint32_t)));

  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // smem: [A_compressed][meta][B_dense]
  uint32_t smem_a_bytes    = tileM * (tileK_elem / 2) * sizeof(itype_t);
  uint32_t smem_meta_bytes = kWgMetaBanks * kMetaStrWords * 4;
  uint32_t smem_b_bytes    = tileK_elem * tileN * sizeof(itype_t);
  uint32_t smem_size       = smem_a_bytes + smem_meta_bytes + smem_b_bytes;

  auto time_start = std::chrono::high_resolution_clock::now();

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, smem_size));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // download result
  std::vector<otype_t> h_C(sizeC);
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t)));

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
