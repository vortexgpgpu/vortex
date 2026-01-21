#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static uint32_t num_tiles = 4;  // Default: 4 back-to-back operations

static void show_usage() {
   std::cout << "VEGETA Pipeline Benchmark Test" << std::endl;
   std::cout << "Tests cycle-accurate pipelining with multiple TILE_GEMM operations." << std::endl;
   std::cout << std::endl;
   std::cout << "Usage: [-n num_tiles] [-h: help]" << std::endl;
   std::cout << "  -n num_tiles: Number of back-to-back TILE_GEMM operations [default: 4]" << std::endl;
   std::cout << std::endl;
   std::cout << "Pipeline Model (α=" << VEGETA_ALPHA << ", β=" << VEGETA_BETA << "):" << std::endl;
   std::cout << "  Single TILE_GEMM latency: " << SINGLE_INSTR_LATENCY << " cycles" << std::endl;
   std::cout << "    WL=" << WL_LATENCY << " + FF=" << FF_LATENCY;
   std::cout << " + FS=" << FS_LATENCY << " + DR=" << DR_LATENCY;
   std::cout << " + REDUCE=" << REDUCE_LATENCY << std::endl;
   std::cout << "  Pipeline initiation interval: " << PIPELINE_II << " cycles" << std::endl;
   std::cout << std::endl;
   std::cout << "Expected cycle counts:" << std::endl;
   std::cout << "  N=1: " << SINGLE_INSTR_LATENCY << " cycles" << std::endl;
   std::cout << "  N=2: " << (SINGLE_INSTR_LATENCY + PIPELINE_II) << " cycles (pipelined)" << std::endl;
   std::cout << "  N=4: " << (SINGLE_INSTR_LATENCY + 3*PIPELINE_II) << " cycles (pipelined)" << std::endl;
   std::cout << "  Without pipelining N=4 would be: " << (4 * SINGLE_INSTR_LATENCY) << " cycles" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      num_tiles = atoi(optarg);
      if (num_tiles < 1) {
        std::cerr << "Error: num_tiles must be >= 1" << std::endl;
        exit(-1);
      }
      break;
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

// CPU reference: C = sum(A[i] * B[i]) for i in 0..num_tiles
static void matmul_cpu(float* C, const std::vector<std::vector<float>>& A_tiles, 
                       const std::vector<std::vector<float>>& B_tiles, uint32_t num) {
  // Initialize C to zero
  std::fill(C, C + TILE_SIZE * TILE_SIZE, 0.0f);
  
  for (uint32_t t = 0; t < num; ++t) {
    for (int m = 0; m < TILE_SIZE; ++m) {
      for (int n = 0; n < TILE_SIZE; ++n) {
        for (int k = 0; k < TILE_SIZE; ++k) {
          C[m * TILE_SIZE + n] += A_tiles[t][m * TILE_SIZE + k] * B_tiles[t][k * TILE_SIZE + n];
        }
      }
    }
  }
}

// Compare floats with ULP tolerance
static bool compare_float(float a, float b, int index, int& errors) {
  union { float f; int32_t i; } fa, fb;
  fa.f = a;
  fb.f = b;
  auto d = std::abs(fa.i - fb.i);
  if (d > FLOAT_ULP) {
    if (errors < 10) {
      printf("*** error: [%d] expected=%.6f, actual=%.6f\n", index, b, a);
    }
    ++errors;
    return false;
  }
  return true;
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  std::srand(42);

  std::cout << "============================================" << std::endl;
  std::cout << "VEGETA Pipeline Benchmark Test" << std::endl;
  std::cout << "============================================" << std::endl;
  std::cout << "Configuration (α=" << VEGETA_ALPHA << ", β=" << VEGETA_BETA << "):" << std::endl;
  std::cout << "  Num TILE_GEMM ops: " << num_tiles << std::endl;
  std::cout << "  Single op latency: " << SINGLE_INSTR_LATENCY << " cycles" << std::endl;
  std::cout << "  Pipeline II: " << PIPELINE_II << " cycles" << std::endl;
  std::cout << std::endl;
  
  // Expected cycles calculation
  uint32_t expected_pipelined = SINGLE_INSTR_LATENCY + (num_tiles - 1) * PIPELINE_II;
  uint32_t expected_non_pipelined = num_tiles * SINGLE_INSTR_LATENCY;
  float speedup = (float)expected_non_pipelined / expected_pipelined;
  
  std::cout << "Expected cycles:" << std::endl;
  std::cout << "  Pipelined: " << expected_pipelined << " cycles" << std::endl;
  std::cout << "  Non-pipelined: " << expected_non_pipelined << " cycles" << std::endl;
  std::cout << "  Expected speedup: " << speedup << "x" << std::endl;
  std::cout << std::endl;

  // Open device
  std::cout << "Opening device..." << std::endl;
  RT_CHECK(vx_dev_open(&device));

  // Calculate buffer sizes
  uint32_t tile_bytes = T_TILE_BYTES;
  uint32_t A_buf_size = num_tiles * tile_bytes;
  uint32_t B_buf_size = num_tiles * tile_bytes;
  uint32_t C_buf_size = tile_bytes;  // Single output tile

  std::cout << "Buffer sizes:" << std::endl;
  std::cout << "  A: " << A_buf_size << " bytes (" << num_tiles << " tiles)" << std::endl;
  std::cout << "  B: " << B_buf_size << " bytes (" << num_tiles << " tiles)" << std::endl;
  std::cout << "  C: " << C_buf_size << " bytes (1 tile)" << std::endl;
  std::cout << std::endl;

  // Allocate device memory
  RT_CHECK(vx_mem_alloc(device, A_buf_size, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, B_buf_size, VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, C_buf_size, VX_MEM_READ_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));
  kernel_arg.num_tiles = num_tiles;

  // Initialize host buffers
  std::vector<std::vector<float>> h_A_tiles(num_tiles);
  std::vector<std::vector<float>> h_B_tiles(num_tiles);
  std::vector<float> h_A_flat(num_tiles * TILE_SIZE * TILE_SIZE);
  std::vector<float> h_B_flat(num_tiles * TILE_SIZE * TILE_SIZE);
  
  for (uint32_t t = 0; t < num_tiles; ++t) {
    h_A_tiles[t].resize(TILE_SIZE * TILE_SIZE);
    h_B_tiles[t].resize(TILE_SIZE * TILE_SIZE);
    
    for (uint32_t i = 0; i < TILE_SIZE * TILE_SIZE; ++i) {
      h_A_tiles[t][i] = static_cast<float>(rand()) / RAND_MAX;
      h_B_tiles[t][i] = static_cast<float>(rand()) / RAND_MAX;
      h_A_flat[t * TILE_SIZE * TILE_SIZE + i] = h_A_tiles[t][i];
      h_B_flat[t * TILE_SIZE * TILE_SIZE + i] = h_B_tiles[t][i];
    }
  }

  std::vector<float> h_C(TILE_SIZE * TILE_SIZE);
  std::vector<float> h_ref(TILE_SIZE * TILE_SIZE);

  // Upload data
  std::cout << "Uploading data..." << std::endl;
  RT_CHECK(vx_copy_to_dev(A_buffer, h_A_flat.data(), 0, A_buf_size));
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B_flat.data(), 0, B_buf_size));
  
  // Initialize C buffer with zeros (for accumulator initial value)
  std::vector<float> h_C_zeros(TILE_SIZE * TILE_SIZE, 0.0f);
  RT_CHECK(vx_copy_to_dev(C_buffer, h_C_zeros.data(), 0, C_buf_size));

  // Upload kernel
  std::cout << "Uploading kernel..." << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // Run kernel
  std::cout << "Executing " << num_tiles << " pipelined TILE_GEMM operations..." << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Download result
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, C_buf_size));

  // CPU reference
  matmul_cpu(h_ref.data(), h_A_tiles, h_B_tiles, num_tiles);

  // Verify
  std::cout << "Verifying result..." << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < TILE_SIZE * TILE_SIZE; ++i) {
    compare_float(h_C[i], h_ref[i], i, errors);
  }

  // Cleanup
  cleanup();

  std::cout << std::endl;
  std::cout << "============================================" << std::endl;
  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "Result verified: PASSED!" << std::endl;
  std::cout << std::endl;
  std::cout << "Pipeline Analysis:" << std::endl;
  std::cout << "  With " << num_tiles << " TILE_GEMM ops" << std::endl;
  std::cout << "  Expected pipelined: " << expected_pipelined << " cycles" << std::endl;
  std::cout << "  Expected speedup vs sequential: " << speedup << "x" << std::endl;
  std::cout << "============================================" << std::endl;

  return 0;
}
