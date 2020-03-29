#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <vortex.h>

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "?")) != -1) {
    switch (c) {
    case '?': {
      printf("Test.\n");
      printf("Usage: [-h: help]\n");
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

int run_test(vx_buffer_h sbuf, vx_buffer_h dbuf, uint32_t address, uint64_t value, int num_blocks) {
  int err;
  int num_failures = 0;

  // write sbuf data
  for (int i = 0; i < 8 * num_blocks; ++i) {
    ((uint64_t*)vx_host_ptr(sbuf))[i] = shuffle(i, value);
  }

  // write buffer to local memory
  err = vx_copy_to_dev(sbuf, address, 64 * num_blocks, 0);
  if (err != 0)
    return -1;

  // read buffer from local memory
  err = vx_copy_from_dev(dbuf, address, 64 * num_blocks, 0);
  if (err != 0)
    return -1;

  // verify result
  for (int i = 0; i < 8 * num_blocks; ++i) {
    auto curr = ((uint64_t*)vx_host_ptr(dbuf))[i];
    auto ref = shuffle(i, value);
    if (curr != ref) {
      printf("error @ %x: actual %ld, expected %ld\n", address + 64 * i, curr, ref);
      ++num_failures;
    }
  }  
  return num_failures;
}

int main(int argc, char *argv[]) {
  int err;
  int num_failures = 0;

  // parse command arguments
  parse_args(argc, argv);

  // open device connection
  vx_device_h device;
  err = vx_dev_open(&device);
  if (err != 0)
    return -1;

  // create source buffer
  vx_buffer_h sbuf;
  err = vx_alloc_shared_mem(device, 4096, &sbuf);
  if (err != 0) {
    vx_dev_close(device);
    return -1;
  }
  
  // create destination buffer
  vx_buffer_h dbuf;
  err = vx_alloc_shared_mem(device, 4096, &dbuf);
  if (err != 0) {
    vx_buf_release(sbuf);
    vx_dev_close(device);
    return -1;
  }

  // run tests
  num_failures += run_test(sbuf, dbuf, 0x10000000, 0x0badf00d00ff00ff, 1);
  num_failures += run_test(sbuf, dbuf, 0x10000000, 0x0badf00d00ff00ff, 2);
  num_failures += run_test(sbuf, dbuf, 0x20000000, 0xff00ff00ff00ff00, 4);
  num_failures += run_test(sbuf, dbuf, 0x20000000, 0x0badf00d40ff40ff, 8);

  // releae buffers
  vx_buf_release(sbuf);
  vx_buf_release(dbuf);

  // close device
  vx_dev_close(device);

  if (0 == num_failures) {
    printf("Test PASSED\n");
  } else {
    printf("Test FAILED\n");
  }

  return num_failures;
}
