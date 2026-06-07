#include <mpi.h>
#include <iostream>
#include <unistd.h>
#include <vortex.h>
#include <vector>
#include <assert.h>
#include "common.h"

#define RT_CHECK(_expr) \
  do { \
    int _ret = _expr; \
    if (0 == _ret) break; \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup(); \
    exit(-1); \
  } while (false)

const char* kernel_file = "kernel.vxbin";
uint32_t count = 0;

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    vx_mem_free(src_buffer);
    vx_mem_free(dst_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

void gen_src_data(std::vector<int>& src_data, uint32_t size) {
  src_data.resize(size);
  for (uint32_t i = 0; i < size; ++i) {
    int value = std::rand();
    src_data[i] = value;
    //std::cout << std::dec << i << ": value=0x" << std::hex << value << std::endl;
  }
}

void gen_ref_data(std::vector<int>& ref_data, const std::vector<int>& src_data, uint32_t size) {
  ref_data.resize(size);
  for (int i = 0; i < (int)size; ++i) {
    int value = src_data.at(i);

    uint32_t samples = size;
    while (samples--) {
      if ((i & 0x1) == 0) {
        value += 1;
      }
    }

    // none taken
    if (i >= 0x7fffffff) {
      value = 0;
    } else {
      value += 2;
    }

    // diverge
    if (i > 1) {
      if (i > 2) {
        value += 6;
      } else {
        value += 5;
      }
    } else {
      if (i > 0) {
        value += 4;
      } else {
        value += 3;
      }
    }

    // all taken
    if (i >= 0) {
      value += 7;
    } else {
      value = 0;
    }

    // loop
    for (int j = 0, n = i; j < n; ++j) {
      value += src_data.at(j);
    }

    // switch
    switch (i) {
    case 0:
      value += 1;
      break;
    case 1:
      value -= 1;
      break;
    case 2:
      value *= 3;
      break;
    case 3:
      value *= 5;
      break;
    default:
      assert(i < (int)size);
      break;
    }

    // select
    value += (i >= 0) ? ((i > 5) ? src_data.at(0) : i) : ((i < 5) ? src_data.at(1) : -i);

    // min/max
	  value += std::min(src_data.at(i), value);
	  value += std::max(src_data.at(i), value);

    ref_data[i] = value;
  }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // parse args
    int c;
    while ((c = getopt(argc, argv, "n:k:h")) != -1) {
        switch (c) {
        case 'n': count = atoi(optarg); break;
        case 'k': kernel_file = optarg; break;
        case 'h': if(rank==0) std::cout<<"Usage: -n count -k kernel\n"; MPI_Finalize(); return 0;
        default: if(rank==0) std::cout<<"Invalid\n"; MPI_Finalize(); return -1;
        }
    }
    if(count==0) count=1;

    std::srand(50);

    // open device
    RT_CHECK(vx_dev_open(&device));

    uint64_t cores, warps, threads;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &cores));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &warps));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &threads));

    uint32_t total_threads = cores*warps*threads;
    uint32_t num_points = count*total_threads;
    uint32_t buf_size = num_points*sizeof(int32_t);

    // allocate full buffers on each rank
    kernel_arg.num_points = num_points;
    std::vector<int32_t> full_src(num_points);
    std::vector<int32_t> full_dst(num_points);

    if(rank==0) {
        for(uint32_t i=0;i<num_points;++i) full_src[i]=std::rand();
    }

    // broadcast full source to all ranks
    MPI_Bcast(full_src.data(), num_points, MPI_INT, 0, MPI_COMM_WORLD);

    // allocate device memory
    RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &src_buffer));
    RT_CHECK(vx_mem_address(src_buffer, &kernel_arg.src_addr));
    RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &dst_buffer));
    RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));

    RT_CHECK(vx_copy_to_dev(src_buffer, full_src.data(), 0, buf_size));

    // upload kernel
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

    // start device
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    RT_CHECK(vx_copy_from_dev(full_dst.data(), dst_buffer, 0, buf_size));

    // gather results back to rank 0 (no-op here, already full_src)
    // verify on rank 0
    if(rank==0) {
        std::vector<int32_t> h_ref(num_points);
        gen_ref_data(h_ref, full_src, num_points);

        int errors=0;
        for(uint32_t i=0;i<num_points;++i){
            if(full_dst[i]!=h_ref[i]){
                std::cout<<"error at result #"<<i
                         <<": actual 0x"<<std::hex<<full_dst[i]
                         <<", expected 0x"<<h_ref[i]<<std::endl;
                errors++;
            }
        }
        if(errors) std::cout<<"Found "<<errors<<" errors! FAILED!\n";
        else std::cout<<"PASSED!\n";
    }

    cleanup();
    MPI_Finalize();
    return 0;
}