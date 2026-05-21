#include "common.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <rvfloats.h>
#include <tensor.h>
#include <tensor_cfg.h>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex.h>

#define MAX_ERRORS 100

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

using namespace vortex;
namespace vt = tensor;

static_assert(vt::mx_scale_format(vt::ITYPE::id), "sgemm_tcu_mx only supports MX input formats");
static_assert((std::is_same<vt::ITYPE, vt::mxint8>::value && std::is_same<vt::OTYPE, vt::int32>::value)
           || (!std::is_same<vt::ITYPE, vt::mxint8>::value && std::is_same<vt::OTYPE, vt::fp32>::value),
              "sgemm_tcu_mx expects mxint8/int32 or floating MX/fp32");

using cfg = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;
using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

static constexpr uint32_t kElemsPerByte = (vt::ITYPE::bits < 8) ? (8 / vt::ITYPE::bits) : 1;
static constexpr uint32_t kScalePack = 4;

static uint8_t read_nibble(const uint8_t *ptr, uint32_t offset) {
  uint8_t value = ptr[offset / 2];
  return (offset & 1) ? (value >> 4) : (value & 0x0f);
}

static uint32_t pack4(const std::vector<uint8_t> &scales, uint32_t start) {
  uint32_t word = 0;
  for (uint32_t i = 0; i < kScalePack; ++i) {
    if ((start + i) < scales.size()) {
      word |= static_cast<uint32_t>(scales[start + i]) << (8 * i);
    }
  }
  return word;
}

static void pack_mx_a_metadata(std::vector<uint32_t> &packed,
                               const std::vector<uint8_t> &scales,
                               uint32_t M,
                               uint32_t K_logical) {
  uint32_t num_tile_rows = M / cfg::tileM;
  uint32_t logical_tileK = cfg::tileK * kElemsPerByte;
  uint32_t num_k_tiles = K_logical / logical_tileK;
  uint32_t scale_blocks_k = K_logical / vt::ITYPE::ele_block;
  uint32_t tile_scale_blocks_k = logical_tileK / vt::ITYPE::ele_block;
  if (tile_scale_blocks_k == 0) {
    tile_scale_blocks_k = 1;
  }

  packed.assign(num_tile_rows * num_k_tiles * NUM_THREADS, 0);
  for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
      std::vector<uint8_t> tile_scales;
      tile_scales.reserve(cfg::tileM * tile_scale_blocks_k);
      uint32_t first_block = (kt * logical_tileK) / vt::ITYPE::ele_block;
      for (uint32_t m = 0; m < cfg::tileM; ++m) {
        uint32_t row = tr * cfg::tileM + m;
        for (uint32_t kb = 0; kb < tile_scale_blocks_k; ++kb) {
          tile_scales.push_back(scales[row * scale_blocks_k + first_block + kb]);
        }
      }
      uint32_t base = (tr * num_k_tiles + kt) * NUM_THREADS;
      for (uint32_t w = 0; w < NUM_THREADS && (w * kScalePack) < tile_scales.size(); ++w) {
        packed[base + w] = pack4(tile_scales, w * kScalePack);
      }
    }
  }
}

static void pack_mx_b_metadata(std::vector<uint32_t> &packed,
                               const std::vector<uint8_t> &scales,
                               uint32_t N,
                               uint32_t K_logical) {
  uint32_t num_tile_cols = N / cfg::tileN;
  uint32_t logical_tileK = cfg::tileK * kElemsPerByte;
  uint32_t num_k_tiles = K_logical / logical_tileK;
  uint32_t tile_scale_blocks_k = logical_tileK / vt::ITYPE::ele_block;
  if (tile_scale_blocks_k == 0) {
    tile_scale_blocks_k = 1;
  }

  packed.assign(num_tile_cols * num_k_tiles * NUM_THREADS, 0);
  for (uint32_t tc = 0; tc < num_tile_cols; ++tc) {
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
      std::vector<uint8_t> tile_scales;
      tile_scales.reserve(cfg::tileN * tile_scale_blocks_k);
      uint32_t first_block = (kt * logical_tileK) / vt::ITYPE::ele_block;
      for (uint32_t n = 0; n < cfg::tileN; ++n) {
        uint32_t col = tc * cfg::tileN + n;
        for (uint32_t kb = 0; kb < tile_scale_blocks_k; ++kb) {
          tile_scales.push_back(scales[(first_block + kb) * N + col]);
        }
      }
      uint32_t base = (tc * num_k_tiles + kt) * NUM_THREADS;
      for (uint32_t w = 0; w < NUM_THREADS && (w * kScalePack) < tile_scales.size(); ++w) {
        packed[base + w] = pack4(tile_scales, w * kScalePack);
      }
    }
  }
}

static float dequantize_mx_value(const itype_t *data,
                                 const std::vector<uint8_t> &scales,
                                 uint32_t offset,
                                 uint32_t scale_index,
                                 float tensor_scale) {
  uint8_t sf = scales[scale_index];
  if constexpr (std::is_same<vt::ITYPE, vt::mxfp8>::value) {
    return bit_cast<float>(rv_mxfp8tof_s(data[offset], sf, 0, nullptr));
  } else if constexpr (std::is_same<vt::ITYPE, vt::mxint8>::value) {
    float scale = std::ldexp(1.0f, static_cast<int32_t>(sf) - 127);
    return (static_cast<float>(data[offset]) / 64.0f) * scale;
  } else {
    uint8_t q = read_nibble(reinterpret_cast<const uint8_t*>(data), offset);
    return bit_cast<float>(rv_nvfp4tof_s(q, sf, 0, nullptr)) * tensor_scale;
  }
}

static int32_t trunc_shift(int32_t value, int32_t shift) {
  if (shift >= 0) {
    return value << shift;
  }
  uint32_t abs_shift = static_cast<uint32_t>(-shift);
  uint32_t mag = value < 0 ? static_cast<uint32_t>(-value) : static_cast<uint32_t>(value);
  int32_t scaled = static_cast<int32_t>(mag >> abs_shift);
  return value < 0 ? -scaled : scaled;
}

static void matmul_cpu(otype_t *C,
                       const itype_t *A,
                       const itype_t *B,
                       const std::vector<uint8_t> &scale_a,
                       const std::vector<uint8_t> &scale_b,
                       float tensor_scale_a,
                       float tensor_scale_b,
                       uint32_t M,
                       uint32_t N,
                       uint32_t K_logical) {
  uint32_t scale_blocks_k = K_logical / vt::ITYPE::ele_block;
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      if constexpr (std::is_same<vt::ITYPE, vt::mxint8>::value) {
        int32_t sum = 0;
        for (uint32_t k = 0; k < K_logical; ++k) {
          uint32_t scale_k = k / vt::ITYPE::ele_block;
          uint8_t sf_a = scale_a[m * scale_blocks_k + scale_k];
          uint8_t sf_b = scale_b[scale_k * N + n];
          int32_t shift = static_cast<int32_t>(sf_a) - 133 + static_cast<int32_t>(sf_b) - 133;
          int32_t product = static_cast<int32_t>(A[m * K_logical + k]) * static_cast<int32_t>(B[n * K_logical + k]);
          sum += trunc_shift(product, shift);
        }
        C[m * N + n] = sum;
      } else {
        float sum = 0.0f;
        for (uint32_t k = 0; k < K_logical; ++k) {
          uint32_t scale_k = k / vt::ITYPE::ele_block;
          auto a = dequantize_mx_value(A, scale_a, m * K_logical + k,
                                       m * scale_blocks_k + scale_k, tensor_scale_a);
          auto b = dequantize_mx_value(B, scale_b, n * K_logical + k,
                                       scale_k * N + n, tensor_scale_b);
          sum += a * b;
        }
        C[m * N + n] = sum;
      }
    }
  }
}

const char *kernel_file = "kernel.vxbin";

uint32_t xm = 32;
uint32_t xn = 32;
uint32_t xk = 32;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h MX_A_buffer = nullptr;
vx_buffer_h MX_B_buffer = nullptr;
#ifdef TCU_MX_TLS
vx_buffer_h A_tensor_scale_buffer = nullptr;
vx_buffer_h B_tensor_scale_buffer = nullptr;
#endif
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex SGEMM TCU MX Test." << std::endl;
  std::cout << "Usage: [-m M] [-n N] [-k K] [-h]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:h")) != -1) {
    switch (c) {
    case 'm': xm = atoi(optarg); break;
    case 'n': xn = atoi(optarg); break;
    case 'k': xk = atoi(optarg); break;
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
    vx_mem_free(A_buffer);
    vx_mem_free(B_buffer);
    vx_mem_free(C_buffer);
    vx_mem_free(MX_A_buffer);
    vx_mem_free(MX_B_buffer);
#ifdef TCU_MX_TLS
    vx_mem_free(A_tensor_scale_buffer);
    vx_mem_free(B_tensor_scale_buffer);
#endif
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_TCU) == 0) {
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

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K_logical = xk;
  uint32_t logical_tileK = cfg::tileK * kElemsPerByte;
  uint32_t K_storage = K_logical / kElemsPerByte;

  if ((M % cfg::tileM) != 0 || (N % cfg::tileN) != 0 || (K_logical % logical_tileK) != 0) {
    std::cout << "Error: M/N/K must be multiples of tile M=" << cfg::tileM
              << " N=" << cfg::tileN << " K=" << logical_tileK << std::endl;
    return -1;
  }
  if ((K_logical % vt::ITYPE::ele_block) != 0) {
    std::cout << "Error: K must be a multiple of MX block size=" << vt::ITYPE::ele_block << std::endl;
    return -1;
  }

  size_t sizeA = M * K_storage;
  size_t sizeB = K_storage * N;
  size_t sizeC = M * N;
  uint32_t grid_dim[2] = {N / cfg::tileN, M / cfg::tileM};
  uint32_t block_dim[2] = {(uint32_t)NT, 1};

  std::cout << "input data type: " << vt::ITYPE::name << " (id=" << vt::ITYPE::id << ")" << std::endl;
  std::cout << "output data type: " << vt::OTYPE::name << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "WMMA Tile Dimension: M=" << cfg::tileM << ", N=" << cfg::tileN
            << ", K(logical)=" << logical_tileK << ", K(storage)=" << cfg::tileK << std::endl;

  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K_storage;
#ifdef TCU_MX_TLS
  kernel_arg.A_tensor_scale_addr = 0;
  kernel_arg.B_tensor_scale_addr = 0;
#endif
  float A_tensor_scale = 1.0f;
  float B_tensor_scale = 1.0f;

  RT_CHECK(vx_mem_alloc(device, sizeA * sizeof(itype_t), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));

  std::vector<float> h_A_dense(M * K_logical);
  std::vector<float> h_B_dense(K_logical * N);
  for (auto &v : h_A_dense) {
    v = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
  }
  for (auto &v : h_B_dense) {
    v = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
  }

  std::vector<itype_t> h_A(sizeA);
  std::vector<itype_t> h_B(sizeB);
  std::vector<uint8_t> scale_a;
  std::vector<uint8_t> scale_b;

  bool ok = false;
  if constexpr (std::is_same<vt::ITYPE, vt::mxfp8>::value) {
    ok = vt::quantize_mxfp8_a_rowmajor(reinterpret_cast<uint8_t*>(h_A.data()), scale_a, h_A_dense.data(), M, K_logical)
      && vt::quantize_mxfp8_b_colmajor(reinterpret_cast<uint8_t*>(h_B.data()), scale_b, h_B_dense.data(), K_logical, N);
  } else if constexpr (std::is_same<vt::ITYPE, vt::mxint8>::value) {
    ok = vt::quantize_mxint8_a_rowmajor(reinterpret_cast<int8_t*>(h_A.data()), scale_a, h_A_dense.data(), M, K_logical)
      && vt::quantize_mxint8_b_colmajor(reinterpret_cast<int8_t*>(h_B.data()), scale_b, h_B_dense.data(), K_logical, N);
  } else {
    ok = vt::quantize_nvfp4_a_rowmajor(reinterpret_cast<uint8_t*>(h_A.data()), scale_a, A_tensor_scale,
                                       h_A_dense.data(), M, K_logical)
      && vt::quantize_nvfp4_b_colmajor(reinterpret_cast<uint8_t*>(h_B.data()), scale_b, B_tensor_scale,
                                       h_B_dense.data(), K_logical, N);
  }
  if (!ok) {
    std::cout << "Error: MX quantization failed!" << std::endl;
    return -1;
  }

  std::vector<uint32_t> h_mx_a;
  std::vector<uint32_t> h_mx_b;
  pack_mx_a_metadata(h_mx_a, scale_a, M, K_logical);
  pack_mx_b_metadata(h_mx_b, scale_b, N, K_logical);

  RT_CHECK(vx_mem_alloc(device, h_mx_a.size() * sizeof(uint32_t), VX_MEM_READ, &MX_A_buffer));
  RT_CHECK(vx_mem_address(MX_A_buffer, &kernel_arg.MX_A_addr));
  RT_CHECK(vx_mem_alloc(device, h_mx_b.size() * sizeof(uint32_t), VX_MEM_READ, &MX_B_buffer));
  RT_CHECK(vx_mem_address(MX_B_buffer, &kernel_arg.MX_B_addr));
#ifdef TCU_MX_TLS
  if constexpr (std::is_same<vt::ITYPE, vt::nvfp4>::value) {
    RT_CHECK(vx_mem_alloc(device, sizeof(float), VX_MEM_READ, &A_tensor_scale_buffer));
    RT_CHECK(vx_mem_address(A_tensor_scale_buffer, &kernel_arg.A_tensor_scale_addr));
    RT_CHECK(vx_mem_alloc(device, sizeof(float), VX_MEM_READ, &B_tensor_scale_buffer));
    RT_CHECK(vx_mem_address(B_tensor_scale_buffer, &kernel_arg.B_tensor_scale_addr));
  }
#endif

  RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, sizeA * sizeof(itype_t)));
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, sizeB * sizeof(itype_t)));
  RT_CHECK(vx_copy_to_dev(MX_A_buffer, h_mx_a.data(), 0, h_mx_a.size() * sizeof(uint32_t)));
  RT_CHECK(vx_copy_to_dev(MX_B_buffer, h_mx_b.data(), 0, h_mx_b.size() * sizeof(uint32_t)));
#ifdef TCU_MX_TLS
  if constexpr (std::is_same<vt::ITYPE, vt::nvfp4>::value) {
    RT_CHECK(vx_copy_to_dev(A_tensor_scale_buffer, &A_tensor_scale, 0, sizeof(float)));
    RT_CHECK(vx_copy_to_dev(B_tensor_scale_buffer, &B_tensor_scale, 0, sizeof(float)));
  }
#endif

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, 0));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<otype_t> h_C(sizeC);
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t)));

  std::vector<otype_t> h_ref(sizeC);
  matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), scale_a, scale_b,
             A_tensor_scale, B_tensor_scale, M, N, K_logical);

  int errors = 0;
  float rel_tol = std::is_same<vt::ITYPE, vt::mxint8>::value ? 0.0f :
                  std::is_same<vt::ITYPE, vt::nvfp4>::value ? 0.25f : 0.05f;
  for (uint32_t i = 0; i < h_ref.size(); ++i) {
    float actual = static_cast<float>(h_C[i]);
    float expected = static_cast<float>(h_ref[i]);
    float diff = std::abs(actual - expected);
    float tol = std::is_same<vt::ITYPE, vt::mxint8>::value ? 0.0f :
                std::max(1.0e-3f, std::abs(expected) * rel_tol);
    if (!std::isfinite(actual) || !std::isfinite(expected) || !std::isfinite(diff) || diff > tol) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=%f, actual=%f, diff=%f, tol=%f\n",
               i, expected, actual, diff, tol);
      }
      ++errors;
    }
  }

  cleanup();
  if (errors != 0) {
    std::cout << "Found " << errors << " / " << sizeC << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
