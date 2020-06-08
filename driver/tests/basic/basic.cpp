#include <iostream>
#include <unistd.h>
#include <vortex.h>
#include "common.h"

int test = -1;

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "t:h?")) != -1) {
    switch (c) {
    case 't': {
      test = atoi(optarg);
    } break;
    case 'h': 
    case '?': {
      std::cout << "Test." << std::endl;
      std::cout << "Usage: [-t testno][-h: help]" << std::endl;
      exit(0);
    } break;
    default:
      exit(-1);
    }
  }
}

uint64_t shuffle(int i, uint64_t value) {
  return (value << i) | (value & ((1 << i)-1));;
}

vx_device_h device = nullptr;
vx_buffer_h sbuf = nullptr;
vx_buffer_h dbuf = nullptr;

void cleanup() {
  if (sbuf) {
    vx_buf_release(sbuf);
  }
  if (dbuf) {
    vx_buf_release(dbuf);
  }
  if (device) {
    vx_dev_close(device);
  }
}

int run_memcopy_test(vx_buffer_h sbuf, 
                     vx_buffer_h dbuf, 
                     uint32_t address, 
                     uint64_t value, 
                     int num_blocks) {
  int errors = 0;

  // write sbuf data
  for (int i = 0; i < (64 * num_blocks) / 8; ++i) {
    ((uint64_t*)vx_host_ptr(sbuf))[i] = shuffle(i, value);
  }

  // write buffer to local memory
  std::cout << "write buffer to local memory" << std::endl;
  RT_CHECK(vx_copy_to_dev(sbuf, address, 64 * num_blocks, 0));

  // read buffer from local memory
  std::cout << "read buffer from local memory" << std::endl;
  RT_CHECK(vx_copy_from_dev(dbuf, address, 64 * num_blocks, 0));

  // verify result
  std::cout << "verify result" << std::endl;
  for (int i = 0; i < (64 * num_blocks) / 8; ++i) {
    auto curr = ((uint64_t*)vx_host_ptr(dbuf))[i];
    auto ref = shuffle(i, value);
    if (curr != ref) {
      std::cout << "error at 0x" << std::hex << (address + 8 * i)
                << ": actual 0x" << curr << ", expected 0x" << ref << std::endl;
      ++errors;
    }
  } 
  
  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  return 0;
}

int run_kernel_test(vx_device_h device, 
                    vx_buffer_h sbuf, 
                    vx_buffer_h dbuf, 
                    const char* program) {
  int errors = 0;

  uint64_t seed = 0x0badf00d40ff40ff;
  
  int src_dev_addr  = DEV_MEM_SRC_ADDR;
  int dest_dev_addr = DEV_MEM_DST_ADDR;
  int num_blocks    = NUM_BLOCKS;

  // write sbuf data
  for (int i = 0; i < (64 * num_blocks) / 8; ++i) {
    ((uint64_t*)vx_host_ptr(sbuf))[i] = shuffle(i, seed);
  }

  // write buffer to local memory
  std::cout << "write buffer to local memory" << std::endl;
  RT_CHECK(vx_copy_to_dev(sbuf, src_dev_addr, 64 * num_blocks, 0));
  
  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, program));

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, -1));

  // flush the caches
  std::cout << "flush the caches" << std::endl;
  RT_CHECK(vx_flush_caches(device, dest_dev_addr, 64 * num_blocks));

  // read buffer from local memory
  std::cout << "read buffer from local memory" << std::endl;
  RT_CHECK(vx_copy_from_dev(dbuf, dest_dev_addr, 64 * num_blocks, 0));

  // verify result
  std::cout << "verify result" << std::endl;
  for (int i = 0; i < (64 * num_blocks) / 8; ++i) {
    auto curr = ((uint64_t*)vx_host_ptr(dbuf))[i];
    auto ref = shuffle(i, seed);
    if (curr != ref) {
      std::cout << "error at 0x" << std::hex << (dest_dev_addr + 8 * i)
                << ": actual 0x" << curr << ", expected 0x" << ref << std::endl;
      ++errors;
    }
  } 
  
  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  return 0;
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  // open device connection
  std::cout << "open device connection" << std::endl;
  vx_device_h device;
  RT_CHECK(vx_dev_open(&device));

  // create source buffer
  std::cout << "create source buffer" << std::endl;
  RT_CHECK(vx_alloc_shared_mem(device, 4096, &sbuf));
  
  // create destination buffer
  std::cout << "create destination buffer" << std::endl;
  RT_CHECK(vx_alloc_shared_mem(device, 4096, &dbuf));

  // run tests  
  /*if (0 == test || -1 == test) {
    std::cout << "run memcopy test" << std::endl;
    RT_CHECK(run_memcopy_test(sbuf, dbuf, DEV_MEM_SRC_ADDR, 0x0badf00d00ff00ff, 1));
    RT_CHECK(run_memcopy_test(sbuf, dbuf, DEV_MEM_SRC_ADDR, 0x0badf00d40ff40ff, 64));
  }*/

  if (1 == test || -1 == test) {
    std::cout << "run kernel test" << std::endl;
    RT_CHECK(run_kernel_test(device, sbuf, dbuf, "kernel.bin"));
  }

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "Test PASSED" << std::endl;

  return 0;
}
