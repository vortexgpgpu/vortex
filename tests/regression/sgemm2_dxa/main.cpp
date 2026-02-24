#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex.h>

#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret)                                              \
      break;                                                    \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

template <typename Type>
class Comparator {};

template <>
class Comparator<float> {
public:
  static const char* type_str() {
    return "float";
  }
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t {
      float f;
      int32_t i;
    };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, a, b);
      }
      return false;
    }
    return true;
  }
};

static void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B, uint32_t n) {
  for (uint32_t row = 0; row < n; ++row) {
    for (uint32_t col = 0; col < n; ++col) {
      TYPE sum(0);
      for (uint32_t k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
      }
      out[row * n + col] = sum;
    }
  }
}

const char* kernel_file = "kernel.vxbin";
uint32_t size = 64;
uint32_t tile_size = 8;
uint32_t chunk_k = 0;
uint32_t mode = 2;
uint32_t verify = 1;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};
constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

static uint32_t pack_meta(uint32_t rank, uint32_t elem_size_enc) {
  return ((rank & ((1u << VX_DXA_DESC_META_DIM_BITS) - 1u)) << VX_DXA_DESC_META_DIM_LSB)
       | ((elem_size_enc & ((1u << VX_DXA_DESC_META_ELEMSZ_BITS) - 1u)) << VX_DXA_DESC_META_ELEMSZ_LSB);
}

static uint32_t pack_2x16(uint32_t lo, uint32_t hi) {
  return ((hi & 0xffffu) << 16) | (lo & 0xffffu);
}

static int dxa_program_desc_2d(vx_device_h dev,
                               uint32_t slot,
                               uint64_t base_addr,
                               uint32_t size0,
                               uint32_t size1,
                               uint32_t stride0_bytes,
                               uint32_t tile0,
                               uint32_t tile1) {
  uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = pack_meta(/*rank=*/2, /*elem_size_enc=*/2);
  int ret = 0;

  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_BASE_LO_OFF, (uint32_t)(base_addr & 0xffffffffu));
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_BASE_HI_OFF, (uint32_t)(base_addr >> 32));
  if (ret) return ret;

  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_SIZE0_OFF, size0);
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_SIZE1_OFF, size1);
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_SIZE2_OFF, 1);
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_SIZE3_OFF, 1);
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_SIZE4_OFF, 1);
  if (ret) return ret;

  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_STRIDE0_OFF, stride0_bytes);
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_STRIDE1_OFF, 0);
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_STRIDE2_OFF, 0);
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_STRIDE3_OFF, 0);
  if (ret) return ret;

  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_META_OFF, meta);
  if (ret) return ret;

  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1);
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_ESTRIDE1_OFF, 1);
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_ESTRIDE2_OFF, 1);
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_ESTRIDE3_OFF, 1);
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_ESTRIDE4_OFF, 1);
  if (ret) return ret;

  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_TILESIZE01_OFF, pack_2x16(tile0, tile1));
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_TILESIZE23_OFF, 0);
  if (ret) return ret;
  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_TILESIZE4_OFF, 0);
  if (ret) return ret;

  ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_CFILL_OFF, 0);
  if (ret) return ret;
  return 0;
}

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-n matrix_size] [-t tile_size] [-c chunk_k] "
               "[-m mode(1=single,2=double)] [-q skip_verify] [-h]\n"
            << "  chunk_k=0 means auto full-K bounded by local memory\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:c:m:k:qh")) != -1) {
    switch (c) {
    case 'n': size = atoi(optarg); break;
    case 't': tile_size = atoi(optarg); break;
    case 'c': chunk_k = atoi(optarg); break;
    case 'm': mode = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'q': verify = 0; break;
    case 'h':
      show_usage();
      exit(0);
      break;
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
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  if ((size % tile_size) != 0) {
    std::cout << "Error: size must be divisible by tile_size\n";
    return -1;
  }
  if (mode != 1 && mode != 2) {
    std::cout << "Error: mode must be 1(single) or 2(double)\n";
    return -1;
  }

  std::srand(50);

  std::cout << "open device connection\n";
  RT_CHECK(vx_dev_open(&device));
  uint64_t isa_flags = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
#ifdef ISA_EXT_DXA
  const uint64_t dxa_isa_bit = (1ull << (32 + ISA_EXT_DXA));
  if ((isa_flags & dxa_isa_bit) == 0) {
    std::cerr << "Error: DXA ISA extension is disabled in runtime CONFIGS.\n";
    cleanup();
    return -1;
  }
#endif

  const uint32_t size_sq = size * size;
  const uint32_t buf_size = size_sq * sizeof(TYPE);
  const uint32_t group_size = tile_size * tile_size;

  std::cout << "data type: " << Comparator<TYPE>::type_str() << "\n";
  std::cout << "matrix size: " << size << "x" << size << "\n";
  std::cout << "mode: " << mode << " (1=single, 2=double)\n";

  uint32_t max_localmem = 0;
  RT_CHECK(vx_check_occupancy(device, group_size, &max_localmem));
  const uint32_t stage_count = (mode == 2) ? 2u : 1u;
  const uint32_t bytes_per_k = stage_count * (2u * tile_size * sizeof(TYPE));
  if (bytes_per_k == 0) {
    std::cout << "Error: invalid bytes_per_k\n";
    cleanup();
    return -1;
  }

  // Full-K by default, bounded by SMEM capacity and optional user cap.
  uint32_t chunk_k_target = size;
  if (chunk_k != 0) {
    chunk_k_target = std::min(chunk_k_target, chunk_k);
  }
  uint32_t chunk_k_cap = max_localmem / bytes_per_k;
  chunk_k_cap = std::max(1u, std::min(chunk_k_cap, size));
  uint32_t chosen_chunk_k = std::min(chunk_k_target, chunk_k_cap);
  while (chosen_chunk_k > 1 && (size % chosen_chunk_k) != 0) {
    --chosen_chunk_k;
  }
  if ((size % chosen_chunk_k) != 0) {
    std::cout << "Error: unable to find valid chunk_k divisor\n";
    cleanup();
    return -1;
  }
  chunk_k = chosen_chunk_k;

  const uint32_t tile_elems_a = tile_size * chunk_k;
  const uint32_t tile_elems_b = chunk_k * tile_size;
  const uint32_t local_mem = stage_count * (tile_elems_a + tile_elems_b) * sizeof(TYPE);

  std::cout << "tile size: " << tile_size
            << ", chunk_k(requested)=" << chunk_k_target
            << ", chunk_k(selected)=" << chunk_k << "\n";
  std::cout << "local memory: " << local_mem << " bytes\n";
  std::cout << "occupancy: max_localmem=" << max_localmem << " bytes\n";
  RT_CHECK(max_localmem < local_mem);

  kernel_arg.grid_dim[0] = size / tile_size;
  kernel_arg.grid_dim[1] = size / tile_size;
  kernel_arg.block_dim[0] = tile_size;
  kernel_arg.block_dim[1] = tile_size;
  kernel_arg.size = size;
  kernel_arg.tile_size = tile_size;
  kernel_arg.chunk_k = chunk_k;
  kernel_arg.mode = mode;

  std::cout << "allocate device memory\n";
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));

  std::vector<TYPE> h_A(size_sq);
  std::vector<TYPE> h_B(size_sq);
  std::vector<TYPE> h_C(size_sq);
  for (uint32_t i = 0; i < size_sq; ++i) {
    h_A[i] = Comparator<TYPE>::generate();
    h_B[i] = Comparator<TYPE>::generate();
  }

  RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, buf_size));
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, buf_size));

  // Descriptor A: dim0=k, dim1=row => A[row, k]
  RT_CHECK(dxa_program_desc_2d(device,
                               kDescA,
                               kernel_arg.A_addr,
                               /*size0=*/size,
                               /*size1=*/size,
                               /*stride0=*/size * sizeof(TYPE),
                               /*tile0=*/chunk_k,
                               /*tile1=*/tile_size));

  // Descriptor B: dim0=col, dim1=k => B[k, col]
  RT_CHECK(dxa_program_desc_2d(device,
                               kDescB,
                               kernel_arg.B_addr,
                               /*size0=*/size,
                               /*size1=*/size,
                               /*stride0=*/size * sizeof(TYPE),
                               /*tile0=*/tile_size,
                               /*tile1=*/chunk_k));

  std::cout << "upload kernel\n";
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device\n";
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, buf_size));

  int errors = 0;
  if (verify) {
    std::vector<TYPE> h_ref(size_sq);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), size);
    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<TYPE>::compare(h_C[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors\n";
    std::cout << "FAILED\n";
    return errors;
  }

  std::cout << "PASSED\n";
  return 0;
}
