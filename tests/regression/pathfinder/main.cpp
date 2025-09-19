#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <vortex.h>
#include "common.h"
#include "VX_types.h"

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
    return rand() % 100;
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
private:
  union Float_t { float f; int i; };
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

const char* kernel_file = "kernel.vxbin";
uint32_t size = 64;

vx_device_h device = nullptr;
vx_buffer_h src0_buffer = nullptr;
vx_buffer_h src1_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n words] [-h: help]" << std::endl;
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
    vx_mem_free(src0_buffer);
    vx_mem_free(src1_buffer);
    vx_mem_free(dst_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));


  // Temporary
  /*size = 8;*/

  // Assignments
  uint32_t num_cols   = size;
  uint32_t num_rows   = size;
  uint32_t num_points = num_cols * num_rows;

  uint32_t wall_buf_size = num_points * sizeof(TYPE);
  uint32_t dst_buf_size = num_cols * sizeof(TYPE);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;

  kernel_arg.num_cols = num_cols;
  kernel_arg.num_points = num_points;
  kernel_arg.num_rows = num_rows;


  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, dst_buf_size, VX_MEM_READ_WRITE, &src0_buffer));
  RT_CHECK(vx_mem_address(src0_buffer, &kernel_arg.src0_addr));
  RT_CHECK(vx_mem_alloc(device, dst_buf_size, VX_MEM_READ_WRITE, &src1_buffer));
  RT_CHECK(vx_mem_address(src1_buffer, &kernel_arg.src1_addr));
  RT_CHECK(vx_mem_alloc(device, dst_buf_size, VX_MEM_READ_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));

  std::cout << "dev_src0=0x" << std::hex << kernel_arg.src0_addr << std::endl;
  std::cout << "dev_src1=0x" << std::hex << kernel_arg.src1_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<TYPE> h_wall(num_points);
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_src1(num_cols);
  std::vector<TYPE> h_dst(num_cols);

  for (uint32_t i = 0; i < num_points; ++i) {
    h_wall[i] = Comparator<TYPE>::generate();
  }

  for(uint32_t i = 0; i < size; i++){
    h_src0[i] = h_wall[num_cols + i];
  }

  for(uint32_t i = 0; i < size; i++){
    h_src1[i] = h_wall[i];
  }


  // Temporary (Debug)
  /*int flat_wall[] = {*/
    /*10, 3, 1, 2, 10, 3, 1, 2,*/
    /*5,  6, 7, 8, 5,  6, 7, 8,*/
    /*9,  4, 2, 1, 9,  4, 2, 1,*/
    /*3,  8, 6, 4, 3,  8, 6, 4,*/
    /*3,  4, 1, 2, 8,  6, 1, 8,*/
    /*2,  7, 5, 9, 3,  9, 9, 8,*/
    /*4,  1, 3, 2, 4,  1, 3, 2,*/
    /*10, 3, 1, 2, 10, 3, 1, 2,*/
  /*};*/

  /*for(uint32_t i = 0; i < num_points; i++){*/
    /*h_wall[i] = flat_wall[i];*/
  /*}*/

  /*for(uint32_t i = 0; i < size; i++){*/
    /*h_src0[i] = flat_wall[num_cols + i];*/
  /*}*/

  /*for(uint32_t i = 0; i < size; i++){*/
    /*h_src1[i] = flat_wall[i];*/
  /*}*/



  // upload source buffer0
  std::cout << "upload source buffer0" << std::endl;
  RT_CHECK(vx_copy_to_dev(src0_buffer, h_src0.data(), 0, dst_buf_size));

  // upload source buffer0
  std::cout << "upload source buffer1" << std::endl;
  RT_CHECK(vx_copy_to_dev(src1_buffer, h_src1.data(), 0, dst_buf_size));

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  uint64_t total_cycles_per_core(0);
  uint64_t total_instrs_per_core(0);
  uint64_t cycles_per_core;
  uint64_t instrs_per_core;

  for(uint32_t k = 0; k < size - 1; k++){

    // start device
    std::cout << "start device" << std::endl;
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    // download destination buffer
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, dst_buf_size));

    // Get Results
    RT_CHECK(vx_mpm_query(device, VX_CSR_MCYCLE, 0, &cycles_per_core));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MINSTRET, 0, &instrs_per_core));
    total_cycles_per_core += cycles_per_core;
    total_instrs_per_core += instrs_per_core;

    /*printf("%d %d\n", cycles_per_core, instrs_per_core);*/

    // Prepare next run
    for(uint32_t i = 0; i < size; i++){
        h_src0[i] = h_wall[ (k+2)*(num_cols) + i];
    }

    for(uint32_t i = 0; i < size; i++){
        h_src1[i] = h_dst[i];
    }

    // upload source buffer0
    std::cout << "upload source buffer0" << std::endl;
    RT_CHECK(vx_copy_to_dev(src0_buffer, h_src0.data(), 0, dst_buf_size));

    // upload source buffer0
    std::cout << "upload source buffer1" << std::endl;
    RT_CHECK(vx_copy_to_dev(src1_buffer, h_src1.data(), 0, dst_buf_size));


    printf("%d\n",k);
  }


  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, dst_buf_size));

  // verify result
  std::cout << "verify result" << std::endl;

  // Run Golden result test
  int *result_s        = (int *) malloc(dst_buf_size);
  int *src             = (int *) malloc(dst_buf_size);
  int *flat_wall_check = (int *) malloc(wall_buf_size);

  for(uint32_t i = 0; i < num_points; i++){
    flat_wall_check[i] = h_wall[i];
  }

  int *result = run(flat_wall_check, result_s, src, num_cols, num_rows, 1);

  // Check for errors
  int errors = 0;
  for (uint32_t i = 0; i < num_cols; ++i) {
    auto ref = result[i];
    auto cur = h_dst[i];
    if (!Comparator<TYPE>::compare(cur, ref, i, errors)) {
      ++errors;
    }
  }
  printf("total_cycles=%ld total_insn=%ld\n", total_cycles_per_core, total_instrs_per_core);

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
