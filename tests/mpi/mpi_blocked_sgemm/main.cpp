#include<mpi.h>
#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex2.h>
#include <cmath>
#include "common.h"
#include <mpi.h>

// Cannon's algorithm sums partial products in block order while the CPU
// reference sums linearly, so FP non-associativity permits a wider ULP margin.
#define FLOAT_ULP 32

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
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
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
    if (A_buffer) vx_buffer_release(A_buffer);
    if (B_buffer) vx_buffer_release(B_buffer);
    if (C_buffer) vx_buffer_release(C_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
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

  MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  // ===============================
  // Create 2D Cartesian Topology
  // ===============================

  int dims[2];
  dims[0] = dims[1] = std::sqrt(world_size);

  if (dims[0] * dims[1] != world_size) {
    if (rank == 0)
      std::cout << "World size must be perfect square\n";
    MPI_Finalize();
    return -1;
  }

  int periods[2] = {1, 1};
  MPI_Comm grid_comm;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

  int coords[2];
  MPI_Cart_coords(grid_comm, rank, 2, coords);

  int row = coords[0];
  int col = coords[1];

  uint32_t block_size = size / dims[0];
  uint32_t block_elems = block_size * block_size;
  uint32_t block_bytes = block_elems * sizeof(TYPE);

  std::vector<TYPE> A_block(block_elems);
  std::vector<TYPE> B_block(block_elems);
  std::vector<TYPE> C_block(block_elems, 0);

  std::vector<TYPE> A_full, B_full;

  if (rank == 0) {
    std::srand(50);
    A_full.resize(size * size);
    B_full.resize(size * size);
    for (uint32_t i = 0; i < size*size; i++) {
      A_full[i] = Comparator<TYPE>::generate();
      B_full[i] = Comparator<TYPE>::generate();
    }
  }

  // ==================================
  // Manual block distribution
  // ==================================

  for (int r = 0; r < world_size; r++) {

    int ccoords[2];
    MPI_Cart_coords(grid_comm, r, 2, ccoords);

    int r_row = ccoords[0];
    int r_col = ccoords[1];

    if (rank == 0) {

      std::vector<TYPE> temp(block_elems);

      for (uint32_t i = 0; i < block_size; i++)
        for (uint32_t j = 0; j < block_size; j++)
          temp[i*block_size+j] =
              A_full[(r_row*block_size+i)*size + (r_col*block_size+j)];

      if (r == 0)
        A_block = temp;
      else
        MPI_Send(temp.data(), block_elems, MPI_FLOAT, r, 0, MPI_COMM_WORLD);

      for (uint32_t i = 0; i < block_size; i++)
        for (uint32_t j = 0; j < block_size; j++)
          temp[i*block_size+j] =
              B_full[(r_row*block_size+i)*size + (r_col*block_size+j)];

      if (r == 0)
        B_block = temp;
      else
        MPI_Send(temp.data(), block_elems, MPI_FLOAT, r, 1, MPI_COMM_WORLD);
    }

    if (rank == r && rank != 0) {
      MPI_Recv(A_block.data(), block_elems, MPI_FLOAT, 0, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(B_block.data(), block_elems, MPI_FLOAT, 0, 1,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // ==============================
  // Cannon Initial Alignment
  // ==============================

  int src, dst;

  MPI_Cart_shift(grid_comm, 1, -row, &src, &dst);
  MPI_Sendrecv_replace(A_block.data(), block_elems,
                       MPI_FLOAT, dst, 0, src, 0,
                       grid_comm, MPI_STATUS_IGNORE);

  MPI_Cart_shift(grid_comm, 0, -col, &src, &dst);
  MPI_Sendrecv_replace(B_block.data(), block_elems,
                       MPI_FLOAT, dst, 1, src, 1,
                       grid_comm, MPI_STATUS_IGNORE);

  // ==============================
  // Main Cannon Loop
  // ==============================

  for (int k = 0; k < dims[0]; k++) {

    // Vortex compute block multiply
    RT_CHECK(vx_device_open(0, &device));

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    RT_CHECK(vx_queue_create(device, &qi, &queue));

    uint64_t num_cores, num_threads;
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

    RT_CHECK(vx_buffer_create(device, block_bytes, VX_MEM_READ, &A_buffer));
    RT_CHECK(vx_buffer_create(device, block_bytes, VX_MEM_READ, &B_buffer));
    RT_CHECK(vx_buffer_create(device, block_bytes, VX_MEM_WRITE, &C_buffer));

    RT_CHECK(vx_buffer_address(A_buffer, &kernel_arg.A_addr));
    RT_CHECK(vx_buffer_address(B_buffer, &kernel_arg.B_addr));
    RT_CHECK(vx_buffer_address(C_buffer, &kernel_arg.C_addr));

    kernel_arg.grid_dim[0] = block_size;
    kernel_arg.grid_dim[1] = block_size;
    kernel_arg.size = block_size;

    RT_CHECK(vx_enqueue_write(queue, A_buffer, 0, A_block.data(), block_bytes, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_write(queue, B_buffer, 0, B_block.data(), block_bytes, 0, nullptr, nullptr));

    RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
    RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

    std::vector<TYPE> temp(block_elems);

    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    {
      vx_launch_info_t li = {};
      li.struct_size  = sizeof(li);
      li.kernel       = kernel;
      li.args_host    = &kernel_arg;
      li.args_size    = sizeof(kernel_arg);
      li.ndim         = 1;
      li.grid_dim[0]  = (uint32_t)num_cores;
      li.block_dim[0] = (uint32_t)num_threads;
      RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
    }

    RT_CHECK(vx_enqueue_read(queue, temp.data(), C_buffer, 0, block_bytes, 1, &launch_ev, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    for (uint32_t i = 0; i < block_elems; i++)
      C_block[i] += temp[i];

    cleanup();
    device = nullptr;
    A_buffer = B_buffer = C_buffer = nullptr;
    queue = nullptr; module_ = nullptr; kernel = nullptr;

    // shift left A
    MPI_Cart_shift(grid_comm, 1, -1, &src, &dst);
    MPI_Sendrecv_replace(A_block.data(), block_elems,
                         MPI_FLOAT, dst, 2, src, 2,
                         grid_comm, MPI_STATUS_IGNORE);

    // shift up B
    MPI_Cart_shift(grid_comm, 0, -1, &src, &dst);
    MPI_Sendrecv_replace(B_block.data(), block_elems,
                         MPI_FLOAT, dst, 3, src, 3,
                         grid_comm, MPI_STATUS_IGNORE);
  }
// ==============================
// Gather C blocks to rank 0
// ==============================

std::vector<TYPE> C_full;

if (rank == 0)
  C_full.resize(size * size);

if (rank == 0) {

  // place own block
  for (uint32_t i = 0; i < block_size; i++)
    for (uint32_t j = 0; j < block_size; j++)
      C_full[i*size + j] = C_block[i*block_size + j];

  // receive others
  for (int r = 1; r < world_size; r++) {

    int coords_r[2];
    MPI_Cart_coords(grid_comm, r, 2, coords_r);

    std::vector<TYPE> temp(block_elems);

    MPI_Recv(temp.data(), block_elems, MPI_FLOAT,
             r, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int r_row = coords_r[0];
    int r_col = coords_r[1];

    for (uint32_t i = 0; i < block_size; i++)
      for (uint32_t j = 0; j < block_size; j++)
        C_full[(r_row*block_size+i)*size +
               (r_col*block_size+j)]
          = temp[i*block_size+j];
  }

} else {

  MPI_Send(C_block.data(), block_elems,
           MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
}


if (rank == 0) {

  std::vector<TYPE> ref(size * size);

  matmul_cpu(ref.data(),
             A_full.data(),
             B_full.data(),
             size,
             size);

  int errors = 0;

  for (uint32_t i = 0; i < size*size; i++) {
    if (!Comparator<TYPE>::compare(C_full[i],
                                   ref[i],
                                   i,
                                   errors))
      errors++;
  }

  if (errors)
    std::cout << "FAILED with "
              << errors << " errors\n";
  else
    std::cout << "PASSED\n";
}



  MPI_Finalize();
  return 0;
}