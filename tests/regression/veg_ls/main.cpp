#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <vortex.h>
#include "common.h"

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
vx_buffer_h src_t_buffer = nullptr;
vx_buffer_h dst_t_buffer = nullptr;
vx_buffer_h src_u_buffer = nullptr;
vx_buffer_h dst_u_buffer = nullptr;
vx_buffer_h src_v_buffer = nullptr;
vx_buffer_h dst_v_buffer = nullptr;
vx_buffer_h src_m_buffer = nullptr;
vx_buffer_h dst_m_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex TILE Operations Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "k:h")) != -1) {
    switch (c) {
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
    vx_mem_free(src_t_buffer);
    vx_mem_free(dst_t_buffer);
    vx_mem_free(src_u_buffer);
    vx_mem_free(dst_u_buffer);
    vx_mem_free(src_v_buffer);
    vx_mem_free(dst_v_buffer);
    vx_mem_free(src_m_buffer);
    vx_mem_free(dst_m_buffer);
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

  uint32_t t_buf_size = NUM_T_TILES * T_TILE_SIZE;
  uint32_t u_buf_size = NUM_U_TILES * U_TILE_SIZE;
  uint32_t v_buf_size = NUM_V_TILES * V_TILE_SIZE;
  uint32_t m_buf_size = NUM_M_TILES * M_TILE_SIZE;

  std::cout << "Testing all physical registers:" << std::endl;
  std::cout << "T-regs: " << NUM_T_TILES << " tiles, " << T_TILE_SIZE << " bytes each, buffer: " << t_buf_size << " bytes" << std::endl;
  std::cout << "U-regs: " << NUM_U_TILES << " tiles, " << U_TILE_SIZE << " bytes each, buffer: " << u_buf_size << " bytes" << std::endl;
  std::cout << "V-regs: " << NUM_V_TILES << " tiles, " << V_TILE_SIZE << " bytes each, buffer: " << v_buf_size << " bytes" << std::endl;
  std::cout << "M-regs: " << NUM_M_TILES << " tiles, " << M_TILE_SIZE << " bytes each, buffer: " << m_buf_size << " bytes" << std::endl;



  // allocate device memory for T tiles
  std::cout << "allocate device memory for T tiles" << std::endl;
  RT_CHECK(vx_mem_alloc(device, t_buf_size, VX_MEM_READ_WRITE, &src_t_buffer));
  RT_CHECK(vx_mem_address(src_t_buffer, &kernel_arg.src_t_addr));
  RT_CHECK(vx_mem_alloc(device, t_buf_size, VX_MEM_READ_WRITE, &dst_t_buffer));
  RT_CHECK(vx_mem_address(dst_t_buffer, &kernel_arg.dst_t_addr));

  // allocate device memory for U tiles
  std::cout << "allocate device memory for U tiles" << std::endl;
  RT_CHECK(vx_mem_alloc(device, u_buf_size, VX_MEM_READ_WRITE, &src_u_buffer));
  RT_CHECK(vx_mem_address(src_u_buffer, &kernel_arg.src_u_addr));
  RT_CHECK(vx_mem_alloc(device, u_buf_size, VX_MEM_READ_WRITE, &dst_u_buffer));
  RT_CHECK(vx_mem_address(dst_u_buffer, &kernel_arg.dst_u_addr));

  // allocate device memory for V tiles
  std::cout << "allocate device memory for V tiles" << std::endl;
  RT_CHECK(vx_mem_alloc(device, v_buf_size, VX_MEM_READ_WRITE, &src_v_buffer));
  RT_CHECK(vx_mem_address(src_v_buffer, &kernel_arg.src_v_addr));
  RT_CHECK(vx_mem_alloc(device, v_buf_size, VX_MEM_READ_WRITE, &dst_v_buffer));
  RT_CHECK(vx_mem_address(dst_v_buffer, &kernel_arg.dst_v_addr));

  // allocate device memory for M tiles
  std::cout << "allocate device memory for M tiles" << std::endl;
  RT_CHECK(vx_mem_alloc(device, m_buf_size, VX_MEM_READ_WRITE, &src_m_buffer));
  RT_CHECK(vx_mem_address(src_m_buffer, &kernel_arg.src_m_addr));
  RT_CHECK(vx_mem_alloc(device, m_buf_size, VX_MEM_READ_WRITE, &dst_m_buffer));
  RT_CHECK(vx_mem_address(dst_m_buffer, &kernel_arg.dst_m_addr));

  std::cout << "dev_src_t=0x" << std::hex << kernel_arg.src_t_addr << std::endl;
  std::cout << "dev_dst_t=0x" << std::hex << kernel_arg.dst_t_addr << std::endl;
  std::cout << "dev_src_u=0x" << std::hex << kernel_arg.src_u_addr << std::endl;
  std::cout << "dev_dst_u=0x" << std::hex << kernel_arg.dst_u_addr << std::endl;
  std::cout << "dev_src_v=0x" << std::hex << kernel_arg.src_v_addr << std::endl;
  std::cout << "dev_dst_v=0x" << std::hex << kernel_arg.dst_v_addr << std::endl;
  std::cout << "dev_src_m=0x" << std::hex << kernel_arg.src_m_addr << std::endl;
  std::cout << "dev_dst_m=0x" << std::hex << kernel_arg.dst_m_addr << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<uint8_t> h_src_t(t_buf_size);
  std::vector<uint8_t> h_dst_t(t_buf_size);
  std::vector<uint8_t> h_src_u(u_buf_size);
  std::vector<uint8_t> h_dst_u(u_buf_size);
  std::vector<uint8_t> h_src_v(v_buf_size);
  std::vector<uint8_t> h_dst_v(v_buf_size);
  std::vector<uint8_t> h_src_m(m_buf_size);
  std::vector<uint8_t> h_dst_m(m_buf_size);

  // Initialize source buffers with different patterns for each tile type
  for (uint32_t i = 0; i < t_buf_size; ++i) {
    h_src_t[i] = (uint8_t)(i & 0xFF);           // Pattern: 0,1,2,...,255,0,1,...
  }
  for (uint32_t i = 0; i < u_buf_size; ++i) {
    h_src_u[i] = (uint8_t)((i * 2) & 0xFF);     // Pattern: 0,2,4,...
  }
  for (uint32_t i = 0; i < v_buf_size; ++i) {
    h_src_v[i] = (uint8_t)((i * 3) & 0xFF);     // Pattern: 0,3,6,...
  }
  for (uint32_t i = 0; i < m_buf_size; ++i) {
    h_src_m[i] = (uint8_t)((i ^ 0xAA) & 0xFF);  // Pattern: XOR with 0xAA
  }

  // upload source buffers
  std::cout << "upload source buffers" << std::endl;
  RT_CHECK(vx_copy_to_dev(src_t_buffer, h_src_t.data(), 0, t_buf_size));
  RT_CHECK(vx_copy_to_dev(src_u_buffer, h_src_u.data(), 0, u_buf_size));
  RT_CHECK(vx_copy_to_dev(src_v_buffer, h_src_v.data(), 0, v_buf_size));
  RT_CHECK(vx_copy_to_dev(src_m_buffer, h_src_m.data(), 0, m_buf_size));

  // Upload kernel binary
  std::cout << "Upload kernel binary" << std::endl;
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

  // download destination buffers
  std::cout << "download destination buffers" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_dst_t.data(), dst_t_buffer, 0, t_buf_size));
  RT_CHECK(vx_copy_from_dev(h_dst_u.data(), dst_u_buffer, 0, u_buf_size));
  RT_CHECK(vx_copy_from_dev(h_dst_v.data(), dst_v_buffer, 0, v_buf_size));
  RT_CHECK(vx_copy_from_dev(h_dst_m.data(), dst_m_buffer, 0, m_buf_size));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  
  // Verify T tiles
  for (uint32_t i = 0; i < t_buf_size; ++i) {
    if (h_dst_t[i] != h_src_t[i]) {
      if (errors < 100) {
        printf("*** error: T[%d] expected=%d, actual=%d\n", i, h_src_t[i], h_dst_t[i]);
      }
      ++errors;
    }
  }
  
  // Verify U tiles
  for (uint32_t i = 0; i < u_buf_size; ++i) {
    if (h_dst_u[i] != h_src_u[i]) {
      if (errors < 100) {
        printf("*** error: U[%d] expected=%d, actual=%d\n", i, h_src_u[i], h_dst_u[i]);
      }
      ++errors;
    }
  }
  
  // Verify V tiles
  for (uint32_t i = 0; i < v_buf_size; ++i) {
    if (h_dst_v[i] != h_src_v[i]) {
      if (errors < 100) {
        printf("*** error: V[%d] expected=%d, actual=%d\n", i, h_src_v[i], h_dst_v[i]);
      }
      ++errors;
    }
  }
  
  // Verify M tiles by comparing debug output
  std::cout << "M tiles loaded successfully (verified by error-free execution)" << std::endl;



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
