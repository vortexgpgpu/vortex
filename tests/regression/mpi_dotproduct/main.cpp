#include <iostream>
#include <unistd.h>
#include <vector>
#include <mpi.h>
#include <vortex2.h>
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
  static float generate(uint32_t /*idx*/) {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int /*index*/, int errors) {
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
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    if (src0_buffer) vx_buffer_release(src0_buffer);
    if (src1_buffer) vx_buffer_release(src1_buffer);
    if (dst_buffer)  vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
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

  uint32_t local_size = ((uint32_t)rank < rem) ? base + 1 : base;

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
    counts[r] = ((uint32_t)r < rem) ? base + 1 : base;
    displs[r] = r * base + std::min((uint32_t)r, rem);
  }

  std::vector<TYPE> h_src0(local_size);
  std::vector<TYPE> h_src1(local_size);

  MPI_Scatterv(full_src0.data(), counts.data(), displs.data(), MPI_FLOAT,
               h_src0.data(), local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Scatterv(full_src1.data(), counts.data(), displs.data(), MPI_FLOAT,
               h_src1.data(), local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Open device
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  const uint32_t threadsPerBlock = 8;
  const uint32_t blocksPerGrid =
      (local_size + threadsPerBlock - 1) / threadsPerBlock;

  uint32_t buf_size = local_size * sizeof(TYPE);
  uint32_t dst_buf_size = blocksPerGrid * sizeof(TYPE);

  kernel_arg.num_points = local_size;
  kernel_arg.block_dim[0] = threadsPerBlock;
  kernel_arg.grid_dim[0]  = blocksPerGrid;

  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src0_buffer));
  RT_CHECK(vx_buffer_address(src0_buffer, &kernel_arg.src0_addr));

  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src1_buffer));
  RT_CHECK(vx_buffer_address(src1_buffer, &kernel_arg.src1_addr));

  RT_CHECK(vx_buffer_create(device, dst_buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

  RT_CHECK(vx_enqueue_write(queue, src0_buffer, 0, h_src0.data(), buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, src1_buffer, 0, h_src1.data(), buf_size, 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::vector<TYPE> h_dst(blocksPerGrid);

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
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, dst_buf_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

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