#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include "common.h"
#include "testcases.h"
#include "sha256.h"

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

const char* kernel_file = "kernel.bin";
uint32_t testno = 0, nmsg = 0;

vx_device_h device = nullptr;
vx_buffer_h buffer = nullptr;

static void show_usage() {
   std::cout << "Vortex Driver Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n nmsg] [-t testno] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:k:h?")) != -1) {
    switch (c) {
    case 'n':
      nmsg = atoi(optarg);
      break;
    case 't':
      testno = atoi(optarg);
      if (testno >= N_TESTCASES) {
        show_usage();
        exit(-1);
      }
      break;
    case 'k':
      kernel_file = optarg;
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
  if (buffer) {
    vx_buf_release(buffer);
  }
  if (device) {
    vx_dev_close(device);
  }
}

int run_test(const kernel_arg_t& kernel_arg,
             testcase_t *testcase,
             uint32_t digestbuf_size, 
             uint32_t num_points) {
  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, -1));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(buffer, kernel_arg.digest_ptr, digestbuf_size, 0));

  // verify result
  std::cout << "verify result" << std::endl;  
  {
    int errors = 0;
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    for (uint32_t i = 0; i < num_points; ++i) {
      if (memcmp(buf_ptr + i * DIGEST_BYTES, testcase->expected_digest, DIGEST_BYTES)) {
        std::cout << "hash mismatch at " << i << std::endl;
        ++errors;
      }
    }
    if (errors != 0) {
      std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
      std::cout << "FAILED!" << std::endl;
      return 1;  
    }
  }

  return 0;
}

int main(int argc, char *argv[]) {
  size_t value; 
  kernel_arg_t kernel_arg;
  
  // parse command arguments
  parse_args(argc, argv);

  if (!testno) {
    testno = DEFAULT_TESTNO;
  }
  if (!nmsg) {
    nmsg = 1;
  }

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  unsigned max_cores, max_warps, max_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_CORES, &max_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_WARPS, &max_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_THREADS, &max_threads));

  testcase_t *testcase = &testcases[testno];
  // Need to allocate enough space for sha256() to pad the message
  uint32_t padded_msgsize = PADDED_SIZE_BYTES(testcase->msgsize);
  uint32_t num_tasks  = max_cores * max_warps * max_threads;
  uint32_t num_points = nmsg * num_tasks;
  uint32_t msgbuf_size = num_points * padded_msgsize;
  uint32_t digestbuf_size = num_points * DIGEST_BYTES;

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "message buffer size: " << msgbuf_size << " bytes" << std::endl;
  std::cout << "digest buffer size: " << digestbuf_size << " bytes" << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;  

  RT_CHECK(vx_alloc_dev_mem(device, msgbuf_size, &value));
  kernel_arg.msg_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, digestbuf_size, &value));
  kernel_arg.digest_ptr = value;

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.msgsize = testcase->msgsize;
  kernel_arg.nmsg = nmsg;

  std::cout << "dev_src=" << std::hex << kernel_arg.msg_ptr << std::dec << std::endl;
  std::cout << "dev_dst=" << std::hex << kernel_arg.digest_ptr << std::dec << std::endl;
  
  // allocate shared memory  
  std::cout << "allocate shared memory" << std::endl;    
  uint32_t alloc_size = std::max<uint32_t>(msgbuf_size, std::max<uint32_t>(digestbuf_size, sizeof(kernel_arg_t)));
  RT_CHECK(vx_alloc_shared_mem(device, alloc_size, &buffer));
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    auto buf_ptr = (int*)vx_host_ptr(buffer);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(buffer, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }

  // upload message (source) buffer
  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    for (uint32_t i = 0; i < num_points; ++i) {
      memcpy(buf_ptr + i * padded_msgsize, testcase->msg, testcase->msgsize);
    }
  }
  std::cout << "upload message buffer" << std::endl;      
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.msg_ptr, msgbuf_size, 0));

  // clear destination buffer
  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    for (uint32_t i = 0; i < num_points; ++i) {
      memset(buf_ptr + i * DIGEST_BYTES, 0, DIGEST_BYTES);
    }
  }
  std::cout << "clear destination buffer" << std::endl;      
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.digest_ptr, digestbuf_size, 0));  

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, testcase, digestbuf_size, num_points));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}
