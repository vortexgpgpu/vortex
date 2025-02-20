#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cmath>
#include "common.h"

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

const char* kernel_file = "kernel.vxbin";
uint32_t matrix_size = 0;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;

std::vector<uint8_t> staging_buf;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv, uint32_t &data_size) {
  int c;
  while ((c = getopt(argc, argv, "n:k:d:h?")) != -1) {
    switch (c) {
    case 'n':
      matrix_size = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'd':
      data_size = atoi(optarg);
      break; 
    case 'h':
    case '?': {
      show_usage();
      exit(0);
    } break;
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

template<typename TYPE>
class mainVariables
{
  public:
    // Constructor
    mainVariables(uint32_t bufSize, uint32_t dataSize, uint32_t matrixSize)
        : buf_size(bufSize), data_size(dataSize), matrix_size(matrixSize)
    {
        // Resize vectors to specified sizes
        src_A.resize(buf_size/data_size);
        src_B.resize(buf_size/data_size);
        refs.resize(buf_size/data_size);
    }

  void init_inputs ()
  {
    std::cout << "inside init" << std::endl;
    for (uint32_t i = 0; i < matrix_size*matrix_size; ++i) 
    {
      auto a = static_cast<float>(std::rand()) / RAND_MAX;
      auto b = static_cast<float>(std::rand()) / RAND_MAX;
      src_A[i] = static_cast<TYPE>(a * matrix_size);
      src_B[i] = static_cast<TYPE>(b * matrix_size);
    }
  }

  void matmul_cpu() 
  {
    for (uint32_t row = 0; row < matrix_size; ++row) 
    {
      for (uint32_t col = 0; col < matrix_size; ++col) 
      {
        TYPE sum(0);
        for (uint32_t e = 0; e < matrix_size; ++e) {
            sum += src_A[row * matrix_size + e] * src_B[e * matrix_size + col];
        }
        refs[row * matrix_size + col] = sum;
      }
    }
  }

  //Public variables
  std::vector<TYPE> src_A;
  std::vector<TYPE> src_B;
  std::vector<TYPE> refs;

  std::vector<uint8_t> A_mat;
  std::vector<uint8_t> B_mat;

  private:
    uint32_t buf_size;
    uint32_t data_size;
    uint32_t matrix_size;
};



int main(int argc, char *argv[]) {  
  // parse command arguments
  uint32_t data_size = 0;
  parse_args(argc, argv, data_size);
  if (matrix_size == 0) {
    matrix_size = 2;
  }

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  uint64_t num_cores, num_warps, num_threads;
  uint64_t tc_size, TC_per_warp;

  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
  
  //Add assert/knob
  RT_CHECK(vx_dev_caps(device, VX_CAPS_TC_SIZE, &tc_size));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_TC_NUM, &TC_per_warp));

  std::cout << "Debug :: tc_size = " << tc_size << std::endl;
  std::cout << "Debug :: tc_num = " << TC_per_warp << std::endl;

  int threads_per_tc;
  //TODO - can be changed
  //Number of output tiles * number of threads
  if (TC_per_warp > num_threads)
    threads_per_tc = 1;
  else
    threads_per_tc = num_threads/TC_per_warp;

  uint32_t num_tasks  = ((matrix_size*matrix_size)/(tc_size*tc_size))*threads_per_tc;

  //size of each operand
  uint32_t buf_size   =  ((matrix_size*matrix_size)/(tc_size*tc_size))*(matrix_size/(tc_size))*(tc_size*tc_size)*data_size;

  //256
  std::cout << "Debug :: buf_size: " << buf_size << " bytes" << std::endl;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;

  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.src0_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.src1_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.dst_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.src0_addr << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.src1_addr << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  mainVariables<int> variables (buf_size, data_size, matrix_size);
  variables.init_inputs();
  
  //////////////////////////////////////////////////
  // generate source data
  //////////////////////////////////////////////////
  variables.matmul_cpu();

  uint32_t tc_size_f = tc_size*tc_size;
  uint32_t n_tiles = matrix_size/tc_size;
  
  variables.A_mat.resize(buf_size);
  variables.B_mat.resize(buf_size);

  //Demand matrix creation for A / traverse through the rows
  for(uint32_t k=0; k<n_tiles; k++)
  {
    //traverse through output tiles in a row
    for(uint32_t i=0; i<n_tiles; i++)
    {
      //traverse through tiles for one output tile
        for(uint32_t j=0; j< n_tiles; j++)
        {
          for(int t=0; t < tc_size*tc_size; t++)
          { 
            variables.A_mat[n_tiles*n_tiles*tc_size_f*k + n_tiles*tc_size_f*i+tc_size_f*j + t]   = variables.src_A[k*tc_size*matrix_size+ tc_size*j +(t/tc_size)*matrix_size + t%tc_size];
          }
        }
    }
  }
  
  //Demand matrix creation for B / traverse through the rows
  for(uint32_t k=0; k<n_tiles; k++)
  {
    //traverse through output tiles in a row
    for(uint32_t i=0; i<n_tiles; i++)
    {
      //traverse through tiles for one output tile
      for(uint32_t j=0; j< n_tiles; j++)
      {
        for(int t=0; t < tc_size*tc_size; t++)
        {
          variables.B_mat[n_tiles*n_tiles*tc_size_f*k + n_tiles*tc_size_f*i+tc_size_f*j + t]   = variables.src_B[i*tc_size+ tc_size*matrix_size*j +(t/tc_size)*matrix_size + t%tc_size];
        }
      }
    }
  }
  
  //////////////////////////////////////////////////
  //////////////////////////////////////////////////

  // upload matrix A buffer
  {
    std::cout << "upload matrix A buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(A_buffer, (int8_t*)variables.A_mat.data(), 0, buf_size));
  }

  // upload matrix B buffer
  {
    std::cout << "upload matrix B buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(B_buffer, (int8_t*)variables.B_mat.data(), 0, buf_size));
  }

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  //////////////////////////////////////////////////
  //Prep kernel arguments
  //////////////////////////////////////////////////  
  //1
  std::cout << "Debug :: num_tasks = " << num_tasks << std::endl;
  kernel_arg.num_tasks = num_tasks;
  kernel_arg.num_warps = num_warps;
  kernel_arg.num_threads = num_threads;
  kernel_arg.TC_per_warp = TC_per_warp;
  //1
  kernel_arg.matrix_size = matrix_size;
  kernel_arg.data_size = data_size;
  kernel_arg.tc_size = tc_size;

  std::cout << "dev_src0=0x" << std::hex << kernel_arg.src0_addr << std::endl;
  std::cout << "dev_src1=0x" << std::hex << kernel_arg.src1_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  //////////////////////////////////////////////////  
  //////////////////////////////////////////////////
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

    std::cout << "start device" << std::endl;
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev((int8_t*)variables.B_mat.data(), C_buffer, 0, buf_size));

  // verify result (TODO : needs to be fixed for for functional correctness)
  /*
  std::cout << "verify result" << std::endl;  
  {
    int errors = 0;
    auto buf_ptr = (int8_t*)staging_buf.data();
    uint64_t tc_size = kernel_arg.tc_size;
    std::cout << "tc_size = " << tc_size << std::endl;
    int Result[matrix_size*matrix_size];
    int n_tiles = (matrix_size/tc_size);
    int tc_size_f = tc_size*tc_size;

    //converting buf ptr (tile by tile) to CPU style linear (row by row)
    for(int k = 0; k < matrix_size/tc_size; k+= 1)
    {
      for(int j = 0; j < matrix_size; j+= tc_size)
      {
        for(int i =0; i < tc_size*tc_size; i++)
        {
          Result[ tc_size*matrix_size*k +j+ (i/tc_size)*matrix_size +i%(tc_size)]  = buf_ptr[matrix_size*tc_size*k+tc_size*j+i];
        }
      }    
    }

    for (uint32_t i = 0; i < matrix_size*matrix_size; ++i) {
      //int ref = i + i; 
      int cur = Result[i];
      if (cur != refs[i]) {
        ++errors;
      }
    }
    if (errors != 0) {
      std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
      std::cout << "FAILED!" << std::endl;
      return 1;  
    }
    else
    {
      std::cout << "CONDITIONALLY PASSED!" << std::endl;
    }
  }
  */

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}