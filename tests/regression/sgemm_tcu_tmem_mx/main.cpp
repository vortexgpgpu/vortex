#include "common.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <rvfloats.h>
#include <tensor_mx.h>
#include <tensor_cfg.h>
#include <type_traits>
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

using cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, WGMMA_NRC>;
static constexpr uint32_t kTileM = cfg::xtileM;
static constexpr uint32_t kTileN = cfg::xtileN;
using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

static_assert(vt::tcu_output_valid(vt::ITYPE::id, vt::OTYPE::id),
              "TCU output type must be fp32 (int32 for integer inputs)");

static constexpr uint32_t kElemsPerByte = (vt::ITYPE::bits < 8) ? (8 / vt::ITYPE::bits) : 1;
// MX scales per A row / B column of a K step (one tc.mma: fedpK words).
static constexpr uint32_t kSf        = vt::mx_max_fedp_sf(cfg::fedpK);
static constexpr uint32_t kStepElems = cfg::fedpK * 4 * kElemsPerByte;
static_assert(kSf <= 4 * cfg::tcN, "a row's scales fit one fragment group");

static uint8_t read_nibble(const uint8_t *ptr, uint32_t offset) {
  uint8_t value = ptr[offset / 2];
  return (offset & 1) ? (value >> 4) : (value & 0x0f);
}

// TMEM scale image of one CTA (rows [row0, row0 + cta_m), columns
// [col0, col0 + tile_n)), every K step, in tc.cp storage order. Per K step
// (sf_cols TMEM columns from step * sf_cols), as vx_tmem.h lays it out:
// A: lane l, column s/4, byte s%4 = scale s of row l. B: byte
// q = col * SF + s of a dense stream of NT*4-byte rows; row r is M step 1 of
// block r % BLOCKS at group 1 + r / BLOCKS. Scale s of a step covers its K
// elements from s * step_elems / SF.
static std::vector<uint32_t> build_sf_image(const std::vector<uint8_t> &scale_a,
                                            const std::vector<uint8_t> &scale_b,
                                            uint32_t row0, uint32_t col0, uint32_t cta_m,
                                            uint32_t tile_n, uint32_t N, uint32_t K_logical,
                                            uint32_t sf_cols, uint32_t ksteps) {
  constexpr uint32_t NT     = VX_CFG_NUM_THREADS;
  constexpr uint32_t BLOCKS = VX_CFG_NUM_TCU_BLOCKS;
  constexpr uint32_t XM     = cfg::xtileM;
  constexpr uint32_t TCM    = cfg::tcM;
  constexpr uint32_t TCN    = cfg::tcN;
  constexpr uint32_t SF     = kSf;
  const uint32_t step_elems = kStepElems;
  const uint32_t sbk = K_logical / vt::ITYPE::ele_block;
  const uint32_t lanes = BLOCKS * XM;
  const uint32_t cols  = ksteps * sf_cols;
  std::vector<uint32_t> img(size_t(lanes) * cols, 0);   // [lane][col]
  auto put = [&](uint32_t lane, uint32_t col, uint32_t byte, uint8_t v) {
    img[size_t(lane) * cols + col] |= uint32_t(v) << (8 * byte);
  };
  for (uint32_t kk = 0; kk < ksteps; ++kk) {
    uint32_t c0 = kk * sf_cols;
    for (uint32_t s = 0; s < SF; ++s) {
      uint32_t kb = (kk * step_elems + s * step_elems / SF) / vt::ITYPE::ele_block;
      for (uint32_t l = 0; l < cta_m; ++l) {
        put(l, c0 + s / 4, s % 4, scale_a[(row0 + l) * sbk + kb]);
      }
      for (uint32_t c = 0; c < tile_n; ++c) {
        uint32_t q = c * SF + s;
        uint32_t r = q / (NT * 4), o = q % (NT * 4);
        uint32_t w = o / 4;
        uint32_t lane = (r % BLOCKS) * XM + TCM + w / TCN;
        uint32_t col  = c0 + (1 + r / BLOCKS) * TCN + w % TCN;
        put(lane, col, o % 4, scale_b[kb * N + col0 + c]);
      }
    }
  }
  // storage order: group by group, block by block, M step by M step, NT words
  std::vector<uint32_t> out;
  out.reserve(img.size());
  for (uint32_t g = 0; g < cols / TCN; ++g)
    for (uint32_t b = 0; b < BLOCKS; ++b)
      for (uint32_t m = 0; m < cfg::m_steps; ++m)
        for (uint32_t i = 0; i < TCM; ++i)
          for (uint32_t j = 0; j < TCN; ++j)
            out.push_back(img[size_t(b * XM + m * TCM + i) * cols + g * TCN + j]);
  return out;
}

static float dequantize_mx_value(const itype_t *data,
                                 const std::vector<uint8_t> &scales,
                                 uint32_t offset,
                                 uint32_t scale_index,
                                 float tensor_scale) {
  uint8_t sf = scales[scale_index];
  switch (vt::ITYPE::id) {
  case vt::mxfp8::id:
    return bit_cast<float>(rv_mxfp8tof_s(data[offset], sf, 0, nullptr));
  case vt::mxbf8::id:
    return bit_cast<float>(rv_mxbf8tof_s(data[offset], sf, 0, nullptr));
  case vt::mxfp4::id:
    {
      uint8_t q = read_nibble(reinterpret_cast<const uint8_t*>(data), offset);
      return bit_cast<float>(rv_mxfp4tof_s(q, sf, 0, nullptr));
    }
  case vt::nvfp4::id:
    {
      uint8_t q = read_nibble(reinterpret_cast<const uint8_t*>(data), offset);
      return bit_cast<float>(rv_nvfp4tof_s(q, sf, 0, nullptr)) * tensor_scale;
    }
  default:
    std::abort();
  }
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

template <typename FormatT>
static bool quantize_inputs(typename FormatT::dtype *A,
                            typename FormatT::dtype *B,
                            std::vector<uint8_t> &scale_a,
                            std::vector<uint8_t> &scale_b,
                            float &A_tensor_scale,
                            float &B_tensor_scale,
                            const float *A_dense,
                            const float *B_dense,
                            uint32_t M,
                            uint32_t N,
                            uint32_t K_logical) {
  if constexpr (std::is_same<FormatT, vt::nvfp4>::value) {
    return vt::quantize_mx_a_rowmajor<FormatT>(
             reinterpret_cast<uint8_t*>(A), scale_a, A_tensor_scale,
             A_dense, M, K_logical)
        && vt::quantize_mx_b_colmajor<FormatT>(
             reinterpret_cast<uint8_t*>(B), scale_b, B_tensor_scale,
             B_dense, K_logical, N);
  } else {
    return vt::quantize_mx_a_rowmajor<FormatT>(
             A, scale_a, A_dense, M, K_logical)
        && vt::quantize_mx_b_colmajor<FormatT>(
             B, scale_b, B_dense, K_logical, N);
  }
}

const char *kernel_file = "kernel.vxbin";

uint32_t xm = 32;
uint32_t xn = 32;
uint32_t xk = 32;

vx_device_h device = nullptr;
vx_queue_h queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h SF_buffer = nullptr;
#ifdef TCU_MX_TLS
vx_buffer_h A_tensor_scale_buffer = nullptr;
vx_buffer_h B_tensor_scale_buffer = nullptr;
#endif
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex SGEMM TCU TMEM MX Test." << std::endl;
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
    vx_mem_free(SF_buffer);
#ifdef TCU_MX_TLS
    vx_mem_free(A_tensor_scale_buffer);
    vx_mem_free(B_tensor_scale_buffer);
#endif
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_dump_perf(device, stdout);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  RT_CHECK(vx_dev_open(&device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_TCU) == 0) {
    std::cout << "TCU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t NT;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT));
  if (NT != VX_CFG_NUM_THREADS) {
    std::cout << "Error: device warp size (" << NT << ") must match VX_CFG_NUM_THREADS=" << VX_CFG_NUM_THREADS << "!" << std::endl;
    return -1;
  }

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K_logical = xk;
  uint32_t logical_tileK = cfg::tileK * kElemsPerByte;
  uint32_t K_storage = K_logical / kElemsPerByte;

  uint32_t warps_per_cta = VX_CFG_ISSUE_WIDTH;
  uint32_t cta_m = warps_per_cta * kTileM;
  if ((M % cta_m) != 0 || (N % kTileN) != 0 || (K_logical % logical_tileK) != 0) {
    std::cout << "Error: M/N/K must be multiples of CTA M=" << cta_m
              << " N=" << kTileN << " K=" << logical_tileK << std::endl;
    return -1;
  }
  if ((K_logical % vt::ITYPE::ele_block) != 0) {
    std::cout << "Error: K must be a multiple of MX block size=" << vt::ITYPE::ele_block << std::endl;
    return -1;
  }

  size_t sizeA = M * K_storage;
  size_t sizeB = K_storage * N;
  size_t sizeC = M * N;
  uint32_t grid_dim[2] = {N / kTileN, M / cta_m};
  uint32_t block_dim[2] = {warps_per_cta * (uint32_t)NT, 1};

  std::cout << "input data type: " << vt::ITYPE::name << " (id=" << vt::ITYPE::id << ")" << std::endl;
  std::cout << "output data type: " << vt::OTYPE::name << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "MMA Tile Dimension: M=" << kTileM << ", N=" << kTileN
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

  bool ok = quantize_inputs<vt::ITYPE>(
      h_A.data(), h_B.data(), scale_a, scale_b, A_tensor_scale, B_tensor_scale,
      h_A_dense.data(), h_B_dense.data(), M, N, K_logical);
  if (!ok) {
    std::cout << "Error: MX quantization failed!" << std::endl;
    return -1;
  }

  // Per K step: one fragment group of A scales, then the B-scale groups.
  const uint32_t ksteps  = K_storage / (cfg::fedpK * 4);
  const uint32_t sf_rows = (kTileN * kSf + NT * 4 - 1) / (NT * 4);
  const uint32_t sf_cols = (1 + (sf_rows + VX_CFG_NUM_TCU_BLOCKS - 1) / VX_CFG_NUM_TCU_BLOCKS) * cfg::tcN;
  std::vector<uint32_t> h_sf;
  for (uint32_t by = 0; by < grid_dim[1]; ++by) {
    for (uint32_t bx = 0; bx < grid_dim[0]; ++bx) {
      auto img = build_sf_image(scale_a, scale_b, by * cta_m, bx * kTileN, cta_m, kTileN,
                                N, K_logical, sf_cols, ksteps);
      kernel_arg.img_words = img.size();
      h_sf.insert(h_sf.end(), img.begin(), img.end());
    }
  }
  kernel_arg.sf_cols = sf_cols;
  RT_CHECK(vx_mem_alloc(device, h_sf.size() * sizeof(uint32_t), VX_MEM_READ, &SF_buffer));
  RT_CHECK(vx_mem_address(SF_buffer, &kernel_arg.SF_addr));
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
  RT_CHECK(vx_copy_to_dev(SF_buffer, h_sf.data(), 0, h_sf.size() * sizeof(uint32_t)));
#ifdef TCU_MX_TLS
  if constexpr (std::is_same<vt::ITYPE, vt::nvfp4>::value) {
    RT_CHECK(vx_copy_to_dev(A_tensor_scale_buffer, &A_tensor_scale, 0, sizeof(float)));
    RT_CHECK(vx_copy_to_dev(B_tensor_scale_buffer, &B_tensor_scale, 0, sizeof(float)));
  }
#endif

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  auto time_start = std::chrono::high_resolution_clock::now();
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = kernel;
  li.args_host = &kernel_arg;
  li.args_size = sizeof(kernel_arg);
  li.ndim = 2;
  li.grid_dim[0] = grid_dim[0];
  li.grid_dim[1] = grid_dim[1];
  li.block_dim[0] = block_dim[0];
  li.block_dim[1] = block_dim[1];
  li.lmem_size = TMEM_SLOT_BYTES + kernel_arg.img_words * sizeof(uint32_t)
               + (cta_m + kTileN) * cfg::tileK * sizeof(itype_t);
  uint32_t tmem_cols = 32;   // the CTA's allocation: a power of two
  while (tmem_cols < kTileN + ksteps * sf_cols) tmem_cols <<= 1;
  li.tmem_cols = tmem_cols;

  vx_event_h launch_ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  RT_CHECK(vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(launch_ev);
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<otype_t> h_C(sizeC);
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t)));

  std::vector<otype_t> h_ref(sizeC);
  matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), scale_a, scale_b,
             A_tensor_scale, B_tensor_scale, M, N, K_logical);

  int errors = 0;
  float rel_tol = (std::is_same<vt::ITYPE, vt::nvfp4>::value || std::is_same<vt::ITYPE, vt::mxfp4>::value) ? 0.25f : 0.05f;
  for (uint32_t i = 0; i < h_ref.size(); ++i) {
    float actual = static_cast<float>(h_C[i]);
    float expected = static_cast<float>(h_ref[i]);
    float diff = std::abs(actual - expected);
    float tol = std::max(1.0e-3f, std::abs(expected) * rel_tol);
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
    return errors != 0;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
