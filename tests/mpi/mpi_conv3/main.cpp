#include <iostream>
#include <vector>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include <vortex2.h>
#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr) \
   do { \
     int _ret = _expr; \
     if (0 == _ret) break; \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
     cleanup(); \
     MPI_Abort(MPI_COMM_WORLD, -1); \
   } while(false)

template <typename Type>
class Comparator {};

template <>
class Comparator<float> {
public:
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union { float f; int i; } fa, fb;
    fa.f = a; fb.f = b;
    int d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP && errors < 10) {
      printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      return false;
    }
    return d <= FLOAT_ULP;
  }
};

static void convolution_cpu(float *O, float *I, float *W, int width, int height) {
  int paddedWidth = width + 2;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int paddedY = y + 1;
      int paddedX = x + 1;
      float sum = 0.0f;
      for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx)
          sum += I[(paddedY+ky)*paddedWidth + (paddedX+kx)]
                 * W[(ky+1)*3 + (kx+1)];
      O[y*width+x] = sum;
    }
  }
}

const char* kernel_file = "kernel.vxbin";
int size = 32;

vx_device_h device = nullptr;
vx_buffer_h I_buf=nullptr, W_buf=nullptr, O_buf=nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    if (I_buf) vx_buffer_release(I_buf);
    if (W_buf) vx_buffer_release(W_buf);
    if (O_buf) vx_buffer_release(O_buf);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char** argv) {

  MPI_Init(&argc,&argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);

if (rank == 0) {
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "-n", 2) == 0) {
            if (strlen(argv[i]) > 2) {
                // case: -n64
                size = atoi(&argv[i][2]);
            } else if (i + 1 < argc) {
                // case: -n 64
                size = atoi(argv[i + 1]);
            }
        }
    }
}

  MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);

  srand(50);

  int rows_per_rank = (size + world_size - 1)/world_size;
  int start = rank*rows_per_rank;
  int end   = std::min(start+rows_per_rank,size);
  int local_rows = end-start;

  int padded_width = size+2;

  std::vector<float> full_I, full_W, full_O;

  if(rank==0){
    full_I.resize((size+2)*(size+2));
    full_W.resize(9);
    full_O.resize(size*size);

    for(int y=-1;y<size+1;y++)
      for(int x=-1;x<size+1;x++)
        full_I[(y+1)*(size+2)+(x+1)] =
          (x>=0&&x<size&&y>=0&&y<size)
          ? Comparator<float>::generate() : 0;

    for(int i=0;i<9;i++)
      full_W[i] = Comparator<float>::generate();
  }

  std::vector<float> local_I((local_rows+2)*padded_width);
  std::vector<float> local_O(local_rows*size);
  std::vector<float> local_W(9);

  if(rank==0)
    local_W = full_W;

  MPI_Bcast(local_W.data(),9,MPI_FLOAT,0,MPI_COMM_WORLD);

  std::vector<int> sendcounts(world_size), displs(world_size);

  for(int r=0;r<world_size;r++){
    int s=r*rows_per_rank;
    int e=std::min(s+rows_per_rank,size);
    sendcounts[r]=(e-s+2)*padded_width;
    displs[r]=s*padded_width;
  }

  MPI_Scatterv(rank==0?full_I.data():nullptr,
               sendcounts.data(),
               displs.data(),
               MPI_FLOAT,
               local_I.data(),
               (local_rows+2)*padded_width,
               MPI_FLOAT,
               0,MPI_COMM_WORLD);

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  kernel_arg.width=size;
  kernel_arg.grid_dim[0]=size;
  kernel_arg.grid_dim[1]=local_rows;
  kernel_arg.use_lmem=false;

  RT_CHECK(vx_buffer_create(device,local_I.size()*sizeof(float),VX_MEM_READ,&I_buf));
  RT_CHECK(vx_buffer_create(device,9*sizeof(float),VX_MEM_READ,&W_buf));
  RT_CHECK(vx_buffer_create(device,local_O.size()*sizeof(float),VX_MEM_WRITE,&O_buf));

  RT_CHECK(vx_buffer_address(I_buf,&kernel_arg.I_addr));
  RT_CHECK(vx_buffer_address(W_buf,&kernel_arg.W_addr));
  RT_CHECK(vx_buffer_address(O_buf,&kernel_arg.O_addr));

  RT_CHECK(vx_enqueue_write(queue,I_buf,0,local_I.data(),local_I.size()*sizeof(float),0,nullptr,nullptr));
  RT_CHECK(vx_enqueue_write(queue,W_buf,0,local_W.data(),9*sizeof(float),0,nullptr,nullptr));

  RT_CHECK(vx_module_load_file(device,kernel_file,&module_));
  RT_CHECK(vx_module_get_kernel(module_,"main",&kernel));

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

  RT_CHECK(vx_enqueue_read(queue,local_O.data(),O_buf,0,local_O.size()*sizeof(float),1,&launch_ev,&read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  std::vector<int> recvcounts(world_size), recvdispls(world_size);

  for(int r=0;r<world_size;r++){
    int s=r*rows_per_rank;
    int e=std::min(s+rows_per_rank,size);
    recvcounts[r]=(e-s)*size;
    recvdispls[r]=s*size;
  }

  MPI_Gatherv(local_O.data(),
              local_rows*size,
              MPI_FLOAT,
              rank==0?full_O.data():nullptr,
              recvcounts.data(),
              recvdispls.data(),
              MPI_FLOAT,
              0,MPI_COMM_WORLD);

  if(rank==0){
    std::vector<float> ref(size*size);
    convolution_cpu(ref.data(),full_I.data(),full_W.data(),size,size);

    int errors=0;
    for(int i=0;i<size*size;i++)
      if(!Comparator<float>::compare(full_O[i],ref[i],i,errors))
        errors++;

    std::cout<<(errors?"FAILED\n":"PASSED\n");
  }

  cleanup();
  MPI_Finalize();
  return 0;
}