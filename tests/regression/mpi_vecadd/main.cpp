#include <iostream>
#include <unistd.h>
#include <vector>
#include <vortex.h>
#include <mpi.h>
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
class Comparator<int> {
public:
  static const char* type_str() { return "integer"; }
  static int generate(uint32_t idx) { return rand(); }
  static bool compare(int a, int b, int index, int errors) {
    if (a != b && errors < 100) {
      printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      return false;
    }
    return a == b;
  }
};

template <>
class Comparator<float> {
public:
  static const char* type_str() { return "float"; }
  static float generate(uint32_t idx) { return static_cast<float>(rand()) / RAND_MAX; }
  static bool compare(float a, float b, int index, int errors) {
    union { float f; int i; } fa, fb;
    fa.f = a; fb.f = b;
    int d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP && errors < 100) {
      printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      return false;
    }
    return d <= FLOAT_ULP;
  }
};

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
      case 'h': std::cout << "Usage: [-k kernel] [-n size] [-h help]\n"; exit(0);
      default: std::cout << "Usage: [-k kernel] [-n size] [-h help]\n"; exit(-1);
    }
  }
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::cout << "rank = " << rank << ", world_size = " << world_size << "\n";
  if (rank == 0) parse_args(argc, argv);
  MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  // Rank 0 generates full input arrays
  std::vector<TYPE> full_src0, full_src1;
  if (rank == 0) {
    std::srand(50);
    full_src0.resize(size);
    full_src1.resize(size);
    for (uint32_t i = 0; i < size; i++) {
      full_src0[i] = Comparator<TYPE>::generate(i);
      full_src1[i] = Comparator<TYPE>::generate(i);
    }
  }

  // Compute local chunk
  uint32_t chunk = (size + world_size - 1) / world_size; // ceil div
  uint32_t start = rank * chunk;
  uint32_t end   = std::min(start + chunk, size);
  uint32_t num_points = end - start;

  // Local buffers
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_src1(num_points);
  std::vector<TYPE> h_dst(num_points);

  // Scatter inputs
  std::vector<int> recvcounts(world_size), displs(world_size);
  for (int i = 0; i < world_size; i++) {
    int s = i * chunk;
    int e = std::min(s + chunk, size);
    recvcounts[i] = e - s;
    displs[i]     = s;
  }

  MPI_Scatterv(full_src0.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
               h_src0.data(), num_points, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Scatterv(full_src1.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
               h_src1.data(), num_points, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Open device
  RT_CHECK(vx_dev_open(&device));

  uint32_t buf_size = num_points * sizeof(TYPE);
  kernel_arg.num_points = num_points;

  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &src0_buffer));
  RT_CHECK(vx_mem_address(src0_buffer, &kernel_arg.src0_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &src1_buffer));
  RT_CHECK(vx_mem_address(src1_buffer, &kernel_arg.src1_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));

  RT_CHECK(vx_copy_to_dev(src0_buffer, h_src0.data(), 0, buf_size));
  RT_CHECK(vx_copy_to_dev(src1_buffer, h_src1.data(), 0, buf_size));

  std::cout << "Rank: " << rank << "- Upload kernel binary" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // Run kernel
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, buf_size));

  // Gather results
  std::vector<TYPE> full_dst;
  if (rank == 0) full_dst.resize(size);

  MPI_Gatherv(h_dst.data(), num_points, MPI_FLOAT,
              full_dst.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
              0, MPI_COMM_WORLD);

  // Verify (rank 0)
  if (rank == 0) {
    int errors = 0;
    for (uint32_t i = 0; i < size; i++) {
      auto ref = full_src0[i] + full_src1[i];
      auto cur = full_dst[i];
      if (!Comparator<TYPE>::compare(cur, ref, i, errors)) errors++;
    }
    if (errors) std::cout << "Found " << errors << " errors!\nFAILED!\n";
    else std::cout << "PASSED!\n";
  }

  cleanup();
  MPI_Finalize();
  return 0;
}
