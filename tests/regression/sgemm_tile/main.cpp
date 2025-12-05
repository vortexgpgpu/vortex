#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();		                                              \
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

static void show_usage() {
   std::cout << "Vortex SGEMM TILE Test (16x16 TGEMM)." << std::endl;
   std::cout << "Usage: [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "h")) != -1) {
    switch (c) {
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

// CPU reference: C = A × B for 16x16 matrices
static void matmul_cpu(float* C, const float* A, const float* B) {
  for (int m = 0; m < TILE_SIZE; ++m) {
    for (int n = 0; n < TILE_SIZE; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < TILE_SIZE; ++k) {
        sum += A[m * TILE_SIZE + k] * B[k * TILE_SIZE + n];
      }
      C[m * TILE_SIZE + n] = sum;
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
    if (errors < 100) {
      printf("*** error: [%d] expected=%.6f, actual=%.6f\n", index, b, a);
    }
    ++errors;
    return false;
  }
  return true;
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint32_t num_elements = TILE_SIZE * TILE_SIZE;  // 256 elements
  uint32_t buf_size = T_TILE_BYTES;               // 1KB

  std::cout << "SGEMM TILE Test: " << TILE_SIZE << "x" << TILE_SIZE << " matrices" << std::endl;
  std::cout << "Buffer size: " << buf_size << " bytes (" << num_elements << " fp32 elements)" << std::endl;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));

  std::cout << "dev_A=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "dev_B=0x" << std::hex << kernel_arg.B_addr << std::endl;
  std::cout << "dev_C=0x" << std::hex << kernel_arg.C_addr << std::dec << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<float> h_A(num_elements);
  std::vector<float> h_B(num_elements);
  std::vector<float> h_C(num_elements);
  std::vector<float> h_ref(num_elements);

  // Initialize with random values
  for (uint32_t i = 0; i < num_elements; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // upload source buffers
  std::cout << "upload source buffers" << std::endl;
  RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, buf_size));
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, buf_size));

  // upload kernel binary
  std::cout << "upload kernel binary" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download result
  std::cout << "download result" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, buf_size));

  // compute CPU reference
  std::cout << "verify result" << std::endl;
  matmul_cpu(h_ref.data(), h_A.data(), h_B.data());

  // verify result
  int errors = 0;
  for (uint32_t i = 0; i < num_elements; ++i) {
    compare_float(h_C[i], h_ref[i], i, errors);
  }

  // write matrices to output file
  std::cout << "writing matrices to output file" << std::endl;
  std::ofstream output_file("matrices_output.txt");
  if (output_file.is_open()) {
    output_file << "Matrix A (" << TILE_SIZE << "x" << TILE_SIZE << "):\n";
    for (int i = 0; i < TILE_SIZE; ++i) {
      for (int j = 0; j < TILE_SIZE; ++j) {
        output_file << h_A[i * TILE_SIZE + j];
        if (j < TILE_SIZE - 1) output_file << " ";
      }
      output_file << "\n";
    }
    output_file << "\n";

    output_file << "Matrix B (" << TILE_SIZE << "x" << TILE_SIZE << "):\n";
    for (int i = 0; i < TILE_SIZE; ++i) {
      for (int j = 0; j < TILE_SIZE; ++j) {
        output_file << h_B[i * TILE_SIZE + j];
        if (j < TILE_SIZE - 1) output_file << " ";
      }
      output_file << "\n";
    }
    output_file << "\n";

    output_file << "Matrix C (Result, " << TILE_SIZE << "x" << TILE_SIZE << "):\n";
    for (int i = 0; i < TILE_SIZE; ++i) {
      for (int j = 0; j < TILE_SIZE; ++j) {
        output_file << h_C[i * TILE_SIZE + j];
        if (j < TILE_SIZE - 1) output_file << " ";
      }
      output_file << "\n";
    }

    output_file.close();
    std::cout << "Matrices written to 'matrices_output.txt'" << std::endl;
  } else {
    std::cerr << "Error: Unable to open output file" << std::endl;
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
