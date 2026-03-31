#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <rvfloats.h>
#include <string.h>
#include <tensor.h>
#include <tensor_cfg.h>
#include <type_traits>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex.h>

#ifndef FLOAT_ULP
#define FLOAT_ULP 6
#endif
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

using cfg = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;
using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

static constexpr bool kIsMxFp8 = std::is_same_v<vt::ITYPE, vt::mxfp8>;
static constexpr bool kIsMxInt8 = std::is_same_v<vt::ITYPE, vt::mxint8>;
static constexpr bool kIsNvFp4 = std::is_same_v<vt::ITYPE, vt::nvfp4>;

static_assert((kIsMxFp8 && std::is_same_v<vt::OTYPE, vt::fp32>)
           || (kIsMxInt8 && std::is_same_v<vt::OTYPE, vt::int32>)
           || (kIsNvFp4 && std::is_same_v<vt::OTYPE, vt::fp32>),
              "sgemm_tcu_mx supports: mxfp8->fp32, mxint8->int32, nvfp4->fp32");

///////////////////////////////////////////////////////////////////////////////

static inline bool compare_output(otype_t a, otype_t b, int index, int errors) {
  if constexpr (std::is_same_v<otype_t, float>) {
    if (a == 0.0f && b == 0.0f) {
      return true;
    }
    float af = static_cast<float>(a);
    float bf = static_cast<float>(b);
    float denom = std::max(std::abs(bf), 1e-8f);
    float diff = std::abs((af - bf) / denom);
    if (diff < 0.01f) {
      return true;
    }
    if (errors < MAX_ERRORS) {
      printf("*** error: [%d] expected=%f, actual=%f (rel=%f)\n",
             index,
             static_cast<double>(bf),
             static_cast<double>(af),
             static_cast<double>(diff));
    }
    return false;
  } else {
    if (a == b) {
      return true;
    }
    if (errors < MAX_ERRORS) {
      std::cout << "*** error: [" << index << "] expected=" << b << ", actual=" << a << std::endl;
    }
    return false;
  }
}

static inline float dequant_mxfp8(uint8_t value, uint8_t scale) {
  return bit_cast<float>(rv_mxfp8tof_s(value, scale, 0, nullptr));
}

static inline float dequant_nvfp4(uint8_t value, uint8_t scale) {
  return bit_cast<float>(rv_nvfp4tof_s(value & 0x0f, scale, 0, nullptr));
}

static inline int32_t round_shift_rne_host(int32_t value, uint32_t rshift) {
  if (rshift == 0) {
    return value;
  }
  if (rshift >= 31) {
    return 0;
  }
  int32_t mask = (1 << rshift) - 1;
  int32_t half = 1 << (rshift - 1);
  int32_t base = value >> rshift;
  int32_t rem = value & mask;
  if (value < 0 && rem != 0) {
    rem -= (1 << rshift);
  }
  if (rem > half) {
    return base + 1;
  }
  if (rem < -half) {
    return base - 1;
  }
  if (rem == half || rem == -half) {
    return (base & 1) ? (base + (rem > 0 ? 1 : -1)) : base;
  }
  return base;
}

static inline int32_t shl_sat_i32_host(int32_t value, uint32_t lshift) {
  if (lshift >= 31) {
    return (value < 0) ? std::numeric_limits<int32_t>::min()
                       : std::numeric_limits<int32_t>::max();
  }
  int64_t scaled = static_cast<int64_t>(value) << lshift;
  if (scaled > std::numeric_limits<int32_t>::max()) {
    return std::numeric_limits<int32_t>::max();
  }
  if (scaled < std::numeric_limits<int32_t>::min()) {
    return std::numeric_limits<int32_t>::min();
  }
  return static_cast<int32_t>(scaled);
}

static inline int32_t madd_wrap_i32_host(int32_t a, int32_t b, int32_t c) {
  int64_t sum = static_cast<int64_t>(a) * static_cast<int64_t>(b)
              + static_cast<int64_t>(c);
  return static_cast<int32_t>(static_cast<uint32_t>(sum));
}

static inline int32_t mxint8_scaled_i32_host(int8_t q, uint8_t sf) {
  int32_t value = static_cast<int32_t>(q);
  int32_t shift = static_cast<int32_t>(sf) - 133;
  if (shift >= 0) {
    return shl_sat_i32_host(value, static_cast<uint32_t>(shift));
  }
  return round_shift_rne_host(value, static_cast<uint32_t>(-shift));
}

static inline uint32_t logical_k_elems(uint32_t K_storage) {
  if constexpr (vt::ITYPE::bits < 8) {
    return K_storage * (8u / vt::ITYPE::bits);
  }
  return K_storage;
}

static inline uint32_t mx_meta_kblock_index(uint32_t k_elem) {
  // Current WMMA metadata ABI carries one scale-bank per tile-K step.
  // Use the tile-start block index so host reference matches device behavior.
  constexpr uint32_t kLogicalTileK = (vt::ITYPE::bits < 8)
                                    ? (cfg::tileK * (8u / vt::ITYPE::bits))
                                    : cfg::tileK;
  uint32_t tile_start = (k_elem / kLogicalTileK) * kLogicalTileK;
  return tile_start / vt::ITYPE::ele_block;
}

static inline uint32_t mx_scale_slots() {
  return kIsNvFp4 ? 16u : 8u;
}

static inline uint32_t mx_meta_words_per_step() {
  return (mx_scale_slots() / 4u) * 2u;
}

static inline uint32_t pack4(const uint8_t* v) {
  return (static_cast<uint32_t>(v[0]) << 0)
       | (static_cast<uint32_t>(v[1]) << 8)
       | (static_cast<uint32_t>(v[2]) << 16)
       | (static_cast<uint32_t>(v[3]) << 24);
}

static void convert_nvfp4_b_row_to_col_major(uint8_t* dst,
                                              const uint8_t* src,
                                              uint32_t K,
                                              uint32_t N) {
  for (uint32_t col = 0; col < N; ++col) {
    for (uint32_t k = 0; k < K; ++k) {
      uint8_t q = vt::detail::data_accessor_t<vt::nvfp4>::read(src, k * N + col);
      vt::detail::data_accessor_t<vt::nvfp4>::write(dst, col * K + k, q);
    }
  }
}

static void normalize_scales_for_wmma_metadata(std::vector<uint8_t>& a_scales,
                                               std::vector<uint8_t>& b_scales,
                                               uint32_t M,
                                               uint32_t N,
                                               uint32_t KS) {
  uint32_t k_blocks = KS / vt::ITYPE::ele_block;
  uint32_t slots = mx_scale_slots();

  if (cfg::tileM > slots) {
    for (uint32_t tr = 0; tr < M; tr += cfg::tileM) {
      for (uint32_t r = slots; r < cfg::tileM && (tr + r) < M; ++r) {
        uint32_t src_r = r - slots;
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
          a_scales[(tr + r) * k_blocks + kb] = a_scales[(tr + src_r) * k_blocks + kb];
        }
      }
    }
  }

  if (cfg::tileN > slots) {
    for (uint32_t tc = 0; tc < N; tc += cfg::tileN) {
      for (uint32_t c = slots; c < cfg::tileN && (tc + c) < N; ++c) {
        uint32_t src_c = c - slots;
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
          b_scales[kb * N + (tc + c)] = b_scales[kb * N + (tc + src_c)];
        }
      }
    }
  }
}

static void pack_meta_mx(std::vector<uint32_t>& meta,
                         const std::vector<uint8_t>& a_scales,
                         const std::vector<uint8_t>& b_scales,
                         uint32_t M,
                         uint32_t N,
                         uint32_t K_storage,
                         uint32_t KS) {
  uint32_t k_blocks = KS / vt::ITYPE::ele_block;
  uint32_t num_tile_rows = M / cfg::tileM;
  uint32_t num_tile_cols = N / cfg::tileN;
  uint32_t num_k_steps = K_storage / cfg::tileK;
  uint32_t slots = mx_scale_slots();
  uint32_t words_per_axis = slots / 4;
  uint32_t per_step_words = mx_meta_words_per_step();

  meta.assign(num_tile_rows * num_tile_cols * num_k_steps * per_step_words, 0);

  constexpr uint32_t kSubbyteRatio = (vt::ITYPE::bits < 8) ? (8 / vt::ITYPE::bits) : 1;

  for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
    for (uint32_t tc = 0; tc < num_tile_cols; ++tc) {
      for (uint32_t ks = 0; ks < num_k_steps; ++ks) {
        uint32_t k0 = ks * cfg::tileK * kSubbyteRatio;
        uint32_t kb = k0 / vt::ITYPE::ele_block;

        uint8_t a_pack[16] = {0};
        for (uint32_t i = 0; i < slots; ++i) {
          uint32_t row = tr * cfg::tileM + i;
          if (row < M) {
            a_pack[i] = a_scales[row * k_blocks + kb];
          }
        }

        uint8_t b_pack[16] = {0};
        for (uint32_t i = 0; i < slots; ++i) {
          uint32_t col = tc * cfg::tileN + i;
          if (col < N) {
            b_pack[i] = b_scales[kb * N + col];
          }
        }

        uint32_t base = (((tr * num_tile_cols + tc) * num_k_steps) + ks) * per_step_words;
        for (uint32_t w = 0; w < words_per_axis; ++w) {
          meta[base + w] = pack4(&a_pack[w * 4]);
        }
        for (uint32_t w = 0; w < words_per_axis; ++w) {
          meta[base + words_per_axis + w] = pack4(&b_pack[w * 4]);
        }
      }
    }
  }
}

static void matmul_cpu_reference(otype_t* C,
                                 const itype_t* A,
                                 const itype_t* B,
                                 const std::vector<uint8_t>& a_scales,
                                 const std::vector<uint8_t>& b_scales,
                                 float a_tensor_scale,
                                 float b_tensor_scale,
                                 uint32_t M,
                                 uint32_t N,
                                 uint32_t K_storage) {
  uint32_t KS = logical_k_elems(K_storage);
  uint32_t k_blocks = KS / vt::ITYPE::ele_block;

  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      if constexpr (kIsMxInt8) {
        int32_t sum = 0;
        for (uint32_t k = 0; k < KS; ++k) {
          uint32_t kb = mx_meta_kblock_index(k);
          uint8_t sf_a = a_scales[m * k_blocks + kb];
          uint8_t sf_b = b_scales[kb * N + n];
          int8_t qa = reinterpret_cast<const int8_t*>(A)[m * KS + k];
          int8_t qb = reinterpret_cast<const int8_t*>(B)[k * N + n];
          int32_t xa = mxint8_scaled_i32_host(qa, sf_a);
          int32_t xb = mxint8_scaled_i32_host(qb, sf_b);
          sum = madd_wrap_i32_host(xa, xb, sum);
        }
        C[m * N + n] = static_cast<otype_t>(sum);
      } else if constexpr (kIsNvFp4) {
        float sum = 0.0f;
        auto A_u8 = reinterpret_cast<const uint8_t*>(A);
        auto B_u8 = reinterpret_cast<const uint8_t*>(B);
        for (uint32_t k = 0; k < KS; ++k) {
          uint32_t kb = mx_meta_kblock_index(k);
          uint8_t sf_a = a_scales[m * k_blocks + kb];
          uint8_t sf_b = b_scales[kb * N + n];
          uint8_t qa = vt::detail::data_accessor_t<vt::nvfp4>::read(A_u8, m * KS + k);
          uint8_t qb = vt::detail::data_accessor_t<vt::nvfp4>::read(B_u8, k * N + n);
          float a = dequant_nvfp4(qa, sf_a);
          float b = dequant_nvfp4(qb, sf_b);
          sum += a * b;
        }
        sum *= (a_tensor_scale * b_tensor_scale);
        C[m * N + n] = static_cast<otype_t>(sum);
      } else {
        float sum = 0.0f;
        auto A_u8 = reinterpret_cast<const uint8_t*>(A);
        auto B_u8 = reinterpret_cast<const uint8_t*>(B);
        for (uint32_t k = 0; k < KS; ++k) {
          uint32_t kb = mx_meta_kblock_index(k);
          uint8_t sf_a = a_scales[m * k_blocks + kb];
          uint8_t sf_b = b_scales[kb * N + n];
          float a = dequant_mxfp8(A_u8[m * KS + k], sf_a);
          float b = dequant_mxfp8(B_u8[k * N + n], sf_b);
          sum += a * b;
        }
        C[m * N + n] = static_cast<otype_t>(sum);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";

uint32_t xm = 32;
uint32_t xn = 32;
uint32_t xk = (cfg::tileK > vt::ITYPE::ele_block) ? cfg::tileK : vt::ITYPE::ele_block;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h meta_mx_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex Sgemm TCU MX Test." << std::endl;
  std::cout << "Usage: [-m M] [-n N] [-k K] [-h]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:h")) != -1) {
    switch (c) {
    case 'm':
      xm = atoi(optarg);
      break;
    case 'n':
      xn = atoi(optarg);
      break;
    case 'k':
      xk = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (A_buffer) vx_mem_free(A_buffer);
    if (B_buffer) vx_mem_free(B_buffer);
    if (C_buffer) vx_mem_free(C_buffer);
    if (meta_mx_buffer) vx_mem_free(meta_mx_buffer);
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  std::srand(50);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  bool has_ext = (isa_flags & VX_ISA_EXT_TCU) != 0;
  if (!has_ext) {
    std::cout << "TCU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t NT;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT));
  if (NT != NUM_THREADS) {
    std::cout << "Error: device warp size (" << NT << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
    cleanup();
    return -1;
  }

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K = xk;
  uint32_t KS = logical_k_elems(K);

  if ((M % cfg::tileM) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileM=" << cfg::tileM << "!" << std::endl;
    cleanup();
    return -1;
  }
  if ((N % cfg::tileN) != 0) {
    std::cout << "Error: N must be a multiple of tensor tileN=" << cfg::tileN << "!" << std::endl;
    cleanup();
    return -1;
  }
  if ((K % cfg::tileK) != 0) {
    std::cout << "Error: K must be a multiple of tensor tileK=" << cfg::tileK << "!" << std::endl;
    cleanup();
    return -1;
  }
  if ((KS % vt::ITYPE::ele_block) != 0) {
    std::cout << "Error: logical K must be a multiple of ITYPE::ele_block=" << vt::ITYPE::ele_block << "!" << std::endl;
    cleanup();
    return -1;
  }

  size_t sizeA = static_cast<size_t>(M) * K;
  size_t sizeB = static_cast<size_t>(K) * N;
  size_t sizeC = static_cast<size_t>(M) * N;
  uint32_t num_tile_rows = M / cfg::tileM;
  uint32_t num_tile_cols = N / cfg::tileN;
  uint32_t num_k_steps = K / cfg::tileK;
  size_t meta_mx_words = static_cast<size_t>(num_tile_rows) * num_tile_cols * num_k_steps * mx_meta_words_per_step();
  uint32_t grid_dim[2] = {N / cfg::tileN, M / cfg::tileM};
  uint32_t block_dim[2] = {(uint32_t)NT, 1};

  std::cout << "input data type: " << vt::ITYPE::name << " (id=" << vt::ITYPE::id << ")" << std::endl;
  std::cout << "output data type: " << vt::OTYPE::name << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "WMMA Core Dimension: M=" << cfg::tcM << ", N=" << cfg::tcN << ", K=" << cfg::tcK << std::endl;
  std::cout << "WMMA Tile Dimension: M=" << cfg::tileM << ", N=" << cfg::tileN << ", K=" << cfg::tileK << std::endl;
  std::cout << "matrix A (storage): " << M << "x" << K << std::endl;
  std::cout << "matrix B (storage): " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;
  std::cout << "logical K elements: " << KS << std::endl;
  std::cout << "meta_mx words: " << meta_mx_words << std::endl;

  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, sizeA * sizeof(itype_t), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));
  RT_CHECK(vx_mem_alloc(device, meta_mx_words * sizeof(uint32_t), VX_MEM_READ, &meta_mx_buffer));
  RT_CHECK(vx_mem_address(meta_mx_buffer, &kernel_arg.meta_mx_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::endl;
  std::cout << "meta_mx_addr=0x" << std::hex << kernel_arg.meta_mx_addr << std::endl;

  std::vector<float> h_A_dense(static_cast<size_t>(M) * KS);
  std::vector<float> h_B_dense(static_cast<size_t>(KS) * N);
  for (uint32_t i = 0; i < h_A_dense.size(); ++i) {
    h_A_dense[i] = 2.0f * (float(rand()) / RAND_MAX) - 1.0f;
  }
  for (uint32_t i = 0; i < h_B_dense.size(); ++i) {
    h_B_dense[i] = 2.0f * (float(rand()) / RAND_MAX) - 1.0f;
  }

  std::vector<itype_t> h_A(sizeA);
  std::vector<itype_t> h_B(sizeB);
  std::vector<uint8_t> h_A_scales;
  std::vector<uint8_t> h_B_scales;
  float a_tensor_scale = 1.0f;
  float b_tensor_scale = 1.0f;

  if constexpr (kIsMxFp8) {
    if (!vt::quantize_mxfp8_a_rowmajor(reinterpret_cast<uint8_t*>(h_A.data()), h_A_scales, h_A_dense.data(), M, KS)) {
      std::cout << "Error: quantize_mxfp8_a_rowmajor failed" << std::endl;
      cleanup();
      return -1;
    }
    if (!vt::quantize_mxfp8_b_rowmajor(reinterpret_cast<uint8_t*>(h_B.data()), h_B_scales, h_B_dense.data(), KS, N)) {
      std::cout << "Error: quantize_mxfp8_b_rowmajor failed" << std::endl;
      cleanup();
      return -1;
    }
  } else if constexpr (kIsMxInt8) {
    if (!vt::quantize_mxint8_a_rowmajor(reinterpret_cast<int8_t*>(h_A.data()), h_A_scales, h_A_dense.data(), M, KS)) {
      std::cout << "Error: quantize_mxint8_a_rowmajor failed" << std::endl;
      cleanup();
      return -1;
    }
    if (!vt::quantize_mxint8_b_rowmajor(reinterpret_cast<int8_t*>(h_B.data()), h_B_scales, h_B_dense.data(), KS, N)) {
      std::cout << "Error: quantize_mxint8_b_rowmajor failed" << std::endl;
      cleanup();
      return -1;
    }
  } else {
    if (!vt::quantize_nvfp4_a_rowmajor(reinterpret_cast<uint8_t*>(h_A.data()), h_A_scales, a_tensor_scale, h_A_dense.data(), M, KS)) {
      std::cout << "Error: quantize_nvfp4_a_rowmajor failed" << std::endl;
      cleanup();
      return -1;
    }
    if (!vt::quantize_nvfp4_b_rowmajor(reinterpret_cast<uint8_t*>(h_B.data()), h_B_scales, b_tensor_scale, h_B_dense.data(), KS, N)) {
      std::cout << "Error: quantize_nvfp4_b_rowmajor failed" << std::endl;
      cleanup();
      return -1;
    }
  }

  normalize_scales_for_wmma_metadata(h_A_scales, h_B_scales, M, N, KS);

  std::vector<uint32_t> h_meta_mx;
  pack_meta_mx(h_meta_mx, h_A_scales, h_B_scales, M, N, K, KS);

  std::cout << "upload matrix A buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, sizeA * sizeof(itype_t)));

  std::cout << "upload matrix B buffer" << std::endl;
  if constexpr (kIsNvFp4) {
    std::vector<uint8_t> h_B_col(sizeB);
    convert_nvfp4_b_row_to_col_major(h_B_col.data(),
                                     reinterpret_cast<const uint8_t*>(h_B.data()),
                                     KS,
                                     N);
    RT_CHECK(vx_copy_to_dev(B_buffer, h_B_col.data(), 0, sizeB * sizeof(itype_t)));
  } else {
    RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, sizeB * sizeof(itype_t)));
  }

  std::cout << "upload matrix MX metadata buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(meta_mx_buffer, h_meta_mx.data(), 0, meta_mx_words * sizeof(uint32_t)));

  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, 0));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<otype_t> h_C(sizeC);
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t)));

  if constexpr (kIsNvFp4) {
    float tensor_mul = a_tensor_scale * b_tensor_scale;
    for (uint32_t i = 0; i < h_C.size(); ++i) {
      h_C[i] *= tensor_mul;
    }
  }

  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<otype_t> h_ref(sizeC);
    matmul_cpu_reference(h_ref.data(), h_A.data(), h_B.data(), h_A_scales, h_B_scales,
                         a_tensor_scale, b_tensor_scale, M, N, K);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!compare_output(h_C[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " / " << sizeC << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
