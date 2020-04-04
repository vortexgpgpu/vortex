#include <iostream>
#include <unistd.h>
#include <vortex.h>

int test = -1;

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

int run_memcopy_test(vx_buffer_h sbuf, 
                    vx_buffer_h dbuf, 
                    uint32_t address, 
                    uint64_t value, 
                    int num_blocks) {
  int ret;
  int errors = 0;

  // write sbuf data
  for (int i = 0; i < 8 * num_blocks; ++i) {
    ((uint64_t*)vx_host_ptr(sbuf))[i] = shuffle(i, value);
  }

  // write buffer to local memory
  std::cout << "write buffer to local memory" << std::endl;
  ret = vx_copy_to_dev(sbuf, address, 64 * num_blocks, 0);
  if (ret != 0)
    return ret;

  // read buffer from local memory
  std::cout << "read buffer from local memory" << std::endl;
  ret = vx_copy_from_dev(dbuf, address, 64 * num_blocks, 0);
  if (ret != 0)
    return ret;

  // verify result
  std::cout << "verify result" << std::endl;
  for (int i = 0; i < 8 * num_blocks; ++i) {
    auto curr = ((uint64_t*)vx_host_ptr(dbuf))[i];
    auto ref = shuffle(i, value);
    if (curr != ref) {
      std::cout << "error @ " << std::hex << (address + 64 * i)
                << ": actual " << curr << ", expected " << ref << std::endl;
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

int run_riscv_test(vx_device_h device, const char* program) {
  int ret;
  
  // upload program
  std::cout << "upload program" << std::endl;  
  ret = vx_upload_kernel_file(device, program);
  if (ret != 0) {
    return ret;  
  }

  // start device
  std::cout << "start device" << std::endl;
  ret = vx_start(device);
  if (ret != 0) {
    return ret;  
  }

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  ret = vx_ready_wait(device, -1);
  if (ret != 0) {
    return ret;  
  }

  return 0;
}

int run_snoop_test(vx_device_h device) {
  int ret;
  
  // upload program
  std::cout << "upload program" << std::endl;  
  ret = vx_upload_kernel_file(device, "snooping.bin");
  if (ret != 0) {
    return ret;  
  }

  // start device
  std::cout << "start device" << std::endl;
  ret = vx_start(device);
  if (ret != 0) {
    return ret;  
  }

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  ret = vx_ready_wait(device, -1);
  if (ret != 0) {
    return ret;  
  }

  // flush the caches
  std::cout << "flush the caches" << std::endl;
  ret = vx_flush_caches(device, 0x10000000, 64*8);
  if (ret != 0) {
    return ret;  
  }

  return 0;
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

int main(int argc, char *argv[]) {
  int ret;

  // parse command arguments
  parse_args(argc, argv);

  // open device connection
  std::cout << "open device connection" << std::endl;
  vx_device_h device;
  ret = vx_dev_open(&device);
  if (ret != 0)
    return ret;

  // create source buffer
  std::cout << "create source buffer" << std::endl;
  ret = vx_alloc_shared_mem(device, 4096, &sbuf);
  if (ret != 0) {
    cleanup();
    return ret;
  }
  
  // create destination buffer
  std::cout << "create destination buffer" << std::endl;
  ret = vx_alloc_shared_mem(device, 4096, &dbuf);
  if (ret != 0) {
    cleanup();
    return ret;
  }

  // run tests  
  if (0 == test || -1 == test) {
    std::cout << "run memcopy test" << std::endl;

    ret = run_memcopy_test(sbuf, dbuf, 0x10000000, 0x0badf00d00ff00ff, 1);
    if (ret != 0) {
      cleanup();
      return ret;
    }

    ret = run_memcopy_test(sbuf, dbuf, 0x20000000, 0x0badf00d40ff40ff, 8);
    if (ret != 0) {
      cleanup();
      return ret;
    }
  }

  if (1 == test || -1 == test) {
    std::cout << "run riscv-lw test" << std::endl;
    ret = run_riscv_test(device, "rv32ui-p-lw.bin");
    if (ret != 0) {
      cleanup();
      return ret;
    }
  }

  if (2 == test || -1 == test) {
    std::cout << "run riscv-sw test" << std::endl;
    ret = run_riscv_test(device, "rv32ui-p-sw.bin");
    if (ret != 0) {
      cleanup();
      return ret;
    }
  }

  if (3 == test || -1 == test) {
    std::cout << "run snoop test" << std::endl;
    ret = run_snoop_test(device);
    if (ret != 0) {
      cleanup();
      return ret;
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "Test PASSED" << std::endl;

  return 0;
}
