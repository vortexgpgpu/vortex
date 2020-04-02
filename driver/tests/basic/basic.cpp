#include <iostream>
#include <unistd.h>
#include <vortex.h>

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "?")) != -1) {
    switch (c) {
    case '?': {
      std::cout << "Test." << std::endl;
      std::cout << "Usage: [-h: help]" << std::endl;
      exit(0);
    } break;
    default:
      exit(-1);
    }
  }
}

uint64_t shuffle(int i, uint64_t value) {
  //return (value << i) | (value & ((1 << i)-1));;
  return 0x0badf00ddeadbeef;
}

int run_test(vx_buffer_h sbuf, 
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
  std::cout << "run tests" << std::endl;
  ret = run_test(sbuf, dbuf, 0x10000000, 0x0badf00d00ff00ff, 1);
  if (ret != 0) {
    cleanup();
    return ret;
  }

  ret = run_test(sbuf, dbuf, 0x10000000, 0x0badf00d00ff00ff, 2);
  if (ret != 0) {
    cleanup();
    return ret;
  }

  ret = run_test(sbuf, dbuf, 0x20000000, 0xff00ff00ff00ff00, 4);
  if (ret != 0) {
    cleanup();
    return ret;
  }

  ret = run_test(sbuf, dbuf, 0x20000000, 0x0badf00d40ff40ff, 8);
  if (ret != 0) {
    cleanup();
    return ret;
  }

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "Test PASSED" << std::endl;

  return 0;
}
