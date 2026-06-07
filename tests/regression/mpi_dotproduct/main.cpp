#include <iostream>
#include <unistd.h>
#include <vector>
#include <mpi.h>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);  \
     cleanup();                                                 \
     MPI_Abort(MPI_COMM_WORLD, -1);                             \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<float> {
public:
  static float generate(uint32_t idx) {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union { float f; int i; } fa, fb;
    fa.f = a; fb.f = b;
    int d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP && errors < 100) {
      printf("*** error: expected=%f, actual=%f\n", b, a);
      return false;
    }
    return d <= FLOAT_ULP;
  }
};

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";
uint32_t size = 16;

vx_device_h device = nullptr;
vx_buffer_h src0_buffer = nullptr;
vx_buffer_h src1_buffer = nullptr;
vx_buffer_h dst_buffer  = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    vx_mem_free(src0_buffer);
    vx_mem_free(src1_buffer);
    vx_mem_free(dst_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
      case 'n': size = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      case 'h': exit(0);
      default: exit(-1);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (rank == 0) parse_args(argc, argv);
  MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  std::srand(50);

  // balanced partitioning
  uint32_t base = size / world_size;
  uint32_t rem  = size % world_size;

  uint32_t local_size = (rank < rem) ? base + 1 : base;
  uint32_t start = rank * base + std::min(rank, (int)rem);

  // full input on rank 0
  std::vector<TYPE> full_src0, full_src1;
  if (rank == 0) {
    full_src0.resize(size);
    full_src1.resize(size);
    for (uint32_t i = 0; i < size; i++) {
      full_src0[i] = Comparator<TYPE>::generate(i);
      full_src1[i] = Comparator<TYPE>::generate(i);
    }
  }

  // Scatter setup
  std::vector<int> counts(world_size), displs(world_size);
  for (int r = 0; r < world_size; r++) {
    counts[r] = (r < rem) ? base + 1 : base;
    displs[r] = r * base + std::min(r, (int)rem);
  }

  std::vector<TYPE> h_src0(local_size);
  std::vector<TYPE> h_src1(local_size);

  MPI_Scatterv(full_src0.data(), counts.data(), displs.data(), MPI_FLOAT,
               h_src0.data(), local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Scatterv(full_src1.data(), counts.data(), displs.data(), MPI_FLOAT,
               h_src1.data(), local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Open device
  RT_CHECK(vx_dev_open(&device));

  const uint32_t threadsPerBlock = 8;
  const uint32_t blocksPerGrid =
      (local_size + threadsPerBlock - 1) / threadsPerBlock;

  uint32_t buf_size = local_size * sizeof(TYPE);
  uint32_t dst_buf_size = blocksPerGrid * sizeof(TYPE);

  kernel_arg.num_points = local_size;
  kernel_arg.block_dim[0] = threadsPerBlock;
  kernel_arg.grid_dim[0]  = blocksPerGrid;

  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &src0_buffer));
  RT_CHECK(vx_mem_address(src0_buffer, &kernel_arg.src0_addr));

  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &src1_buffer));
  RT_CHECK(vx_mem_address(src1_buffer, &kernel_arg.src1_addr));

  RT_CHECK(vx_mem_alloc(device, dst_buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));

  RT_CHECK(vx_copy_to_dev(src0_buffer, h_src0.data(), 0, buf_size));
  RT_CHECK(vx_copy_to_dev(src1_buffer, h_src1.data(), 0, buf_size));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg,
                           sizeof(kernel_arg_t), &args_buffer));

  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<TYPE> h_dst(blocksPerGrid);
  RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, dst_buf_size));

  // Local reduction of block outputs
  TYPE local_sum = 0;
  for (uint32_t i = 0; i < blocksPerGrid; i++)
    local_sum += h_dst[i];

  TYPE global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum,
             1, MPI_FLOAT, MPI_SUM,
             0, MPI_COMM_WORLD);

  // Final verification
  if (rank == 0) {

    TYPE ref = 0;
    for (uint32_t i = 0; i < size; i++)
      ref += full_src0[i] * full_src1[i];

    int errors = 0;
    if (!Comparator<TYPE>::compare(global_sum, ref, 0, errors))
      errors++;

    if (errors)
      std::cout << "FAILED!\n";
    else
      std::cout << "PASSED!\n";
  }

  cleanup();
  MPI_Finalize();
  return 0;
}