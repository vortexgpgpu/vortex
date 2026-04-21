#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cmath>
#include "common.h"
#include <mpi.h>

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<int> {
public:
  static const char* type_str() {
    return "integer";
  }
  static int generate() {
    return rand();
  }
  static bool compare(int a, int b, int index, int errors) {
    if (a != b) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

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
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

static void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B, uint32_t width, uint32_t height) {
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      TYPE sum(0);
      for (uint32_t e = 0; e < width; ++e) {
          sum += A[row * width + e] * B[e * width + col];
      }
      out[row * width + col] = sum;
    }
  }
}

const char* kernel_file = "kernel.vxbin";
uint32_t size = 32;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n size] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
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



////




int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (rank == 0)
    parse_args(argc, argv);

  // Broadcast matrix size
  MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  uint32_t size_sq = size * size;

  // Compute row chunk (ceil division)
  uint32_t rows_per_rank = (size + world_size - 1) / world_size;
  uint32_t row_start = rank * rows_per_rank;
  uint32_t row_end = std::min(row_start + rows_per_rank, size);
  uint32_t local_rows = row_end - row_start;

  uint32_t local_elems = local_rows * size;
  uint32_t local_buf_size = local_elems * sizeof(TYPE);
  uint32_t full_buf_size = size_sq * sizeof(TYPE);

  std::vector<TYPE> h_A_full;
  std::vector<TYPE> h_B(size_sq);
  std::vector<TYPE> h_C_full;

  // Only rank 0 initializes
  if (rank == 0) {
    std::srand(50);
    h_A_full.resize(size_sq);
    h_C_full.resize(size_sq);

    for (uint32_t i = 0; i < size_sq; ++i) {
      h_A_full[i] = Comparator<TYPE>::generate();
      h_B[i] = Comparator<TYPE>::generate();
    }
  }

  // Allocate local A and C
  std::vector<TYPE> h_A_local(local_elems);
  std::vector<TYPE> h_C_local(local_elems);

  // Prepare scatter metadata
  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);

  for (int i = 0; i < world_size; i++) {
    uint32_t rs = i * rows_per_rank;
    uint32_t re = std::min(rs + rows_per_rank, size);
    sendcounts[i] = (re - rs) * size;
    displs[i] = rs * size;
  }

  // Scatter rows of A
  MPI_Scatterv(
      h_A_full.data(),
      sendcounts.data(),
      displs.data(),
      MPI_FLOAT,
      h_A_local.data(),
      local_elems,
      MPI_FLOAT,
      0,
      MPI_COMM_WORLD);

  // Broadcast full B
  MPI_Bcast(
      h_B.data(),
      size_sq,
      MPI_FLOAT,
      0,
      MPI_COMM_WORLD);

  // ============================
  // VORTEX EXECUTION PER RANK
  // ============================

  vx_device_h device;
  vx_buffer_h A_buffer, B_buffer, C_buffer;
  vx_buffer_h krnl_buffer, args_buffer;
  kernel_arg_t kernel_arg = {};

  RT_CHECK(vx_dev_open(&device));

  RT_CHECK(vx_mem_alloc(device, local_buf_size, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_alloc(device, full_buf_size, VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_alloc(device, local_buf_size, VX_MEM_WRITE, &C_buffer));

  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));

  kernel_arg.grid_dim[0] = size;
  kernel_arg.grid_dim[1] = local_rows;
  kernel_arg.size = size;

  RT_CHECK(vx_copy_to_dev(A_buffer, h_A_local.data(), 0, local_buf_size));
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, full_buf_size));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_copy_from_dev(h_C_local.data(), C_buffer, 0, local_buf_size));

  // ============================
  // GATHER RESULTS
  // ============================

  MPI_Gatherv(
      h_C_local.data(),
      local_elems,
      MPI_FLOAT,
      h_C_full.data(),
      sendcounts.data(),
      displs.data(),
      MPI_FLOAT,
      0,
      MPI_COMM_WORLD);

  // ============================
  // VERIFY (Rank 0)
  // ============================

  if (rank == 0) {
    std::vector<TYPE> h_ref(size_sq);
    matmul_cpu(h_ref.data(), h_A_full.data(), h_B.data(), size, size);

    int errors = 0;
    for (uint32_t i = 0; i < size_sq; ++i) {
      if (!Comparator<TYPE>::compare(h_C_full[i], h_ref[i], i, errors))
        errors++;
    }

    if (errors)
      std::cout << "FAILED\n";
    else
      std::cout << "PASSED\n";
  }

  vx_dev_close(device);
  MPI_Finalize();
  return 0;
}