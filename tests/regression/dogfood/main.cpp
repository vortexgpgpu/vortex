#include <iostream>
#include <vector>
#include <unordered_set>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <vortex.h>
#include "testcases.h"
#include "common.h"

///////////////////////////////////////////////////////////////////////////////

TestSuite* testSuite = nullptr;
const char* kernel_file = "kernel.vxbin";
int count = 64;
std::unordered_set<std::string> selected;
std::unordered_set<std::string> excluded;
int testid_s = 0;
int testid_e = 0;
bool stop_on_error = true;

vx_device_h device = nullptr;
vx_buffer_h src0_buffer = nullptr;
vx_buffer_h src1_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-t<name>: select test [-x<name>: exclude test]] [-s<testid>: start test] [-e<testid>: end test]" << std::endl;
   std::cout << "       [-k<kernel>] [-n<words>] [-c] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:x:s:e:k:ch")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
      break;
    case 't':
      selected.insert(optarg);
      break;
    case 'x':
      excluded.insert(optarg);
      break;
    case 's':
      testid_s = atoi(optarg);
      break;
    case 'e':
      testid_e = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'c':
      stop_on_error = false;
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
  if (testSuite) {
    delete testSuite;
  }
  if (device) {
    if (src0_buffer) vx_buffer_release(src0_buffer);
    if (src1_buffer) vx_buffer_release(src1_buffer);
    if (dst_buffer)  vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }

  std::cout << std::dec;

  std::cout << "test ids: " << testid_s << " - " << testid_e << std::endl;
  std::cout << "workitem size: " << count << std::endl;
  std::cout << "using kernel: " << kernel_file << std::endl;

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  int num_tasks = num_cores * num_warps * num_threads;
  int num_points = count * num_tasks;
  size_t buf_size = num_points * sizeof(uint32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.task_size = count;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src0_buffer));
  RT_CHECK(vx_buffer_address(src0_buffer, &kernel_arg.src0_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src1_buffer));
  RT_CHECK(vx_buffer_address(src1_buffer, &kernel_arg.src1_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

  std::cout << "dev_src0=0x" << std::hex << kernel_arg.src0_addr << std::dec << std::endl;
  std::cout << "dev_src1=0x" << std::hex << kernel_arg.src1_addr << std::dec << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::dec << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<uint8_t> src1_buf(buf_size);
  std::vector<uint8_t> src2_buf(buf_size);
  std::vector<uint8_t> dst_buf(buf_size);

  // allocate test suite
  testSuite = new TestSuite(device);
  if (testid_e == 0) {
    testid_e = (testSuite->size() - 1);
  }

  // load kernel module
  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // execute tests
  int errors = 0;
  for (int t = testid_s; t <= testid_e; ++t) {
    auto test = testSuite->get_test(t);
    auto name = test->name();

    if (!selected.empty()) {
      if (selected.count(name) == 0)
        continue;
    }

    if (!excluded.empty()) {
      if (excluded.count(name) != 0)
        continue;
    }

    std::cout << "Test" << t << ": " << name << std::endl;

    // get test arguments
    std::cout << "get test arguments" << std::endl;
    RT_CHECK(test->setup(num_points, (void*)src1_buf.data(), (void*)src2_buf.data()));

    // upload source buffer0
    std::cout << "upload source buffer0" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, src0_buffer, 0, src1_buf.data(), buf_size, 0, nullptr, nullptr));

    // upload source buffer1
    std::cout << "upload source buffer1" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, src1_buffer, 0, src2_buf.data(), buf_size, 0, nullptr, nullptr));

    // clear destination buffer
    std::cout << "clear destination buffer" << std::endl;
    for (int i = 0; i < num_points; ++i) {
      ((uint32_t*)dst_buf.data())[i] = 0xdeadbeef;
    }
    RT_CHECK(vx_enqueue_write(queue, dst_buffer, 0, dst_buf.data(), buf_size, 0, nullptr, nullptr));

    kernel_arg.testid = t;

    // start device
    std::cout << "start device" << std::endl;
    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    {
      uint32_t n = (uint32_t)num_tasks;
      uint32_t grid_dim[1], block_dim[1];
      RT_CHECK(vx_device_max_occupancy_grid(device, 1, &n, grid_dim, block_dim));
      vx_launch_info_t li = {};
      li.struct_size  = sizeof(li);
      li.kernel       = kernel;
      li.args_host    = &kernel_arg;
      li.args_size    = sizeof(kernel_arg);
      li.ndim         = 1;
      li.grid_dim[0]  = grid_dim[0];
      li.block_dim[0] = block_dim[0];
      RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
    }

    // download destination buffer
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_enqueue_read(queue, dst_buf.data(), dst_buffer, 0, buf_size, 1, &launch_ev, &read_ev));

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    // verify destination
    std::cout << "verify test result" << std::endl;
    int err = test->verify(num_points, dst_buf.data(), src1_buf.data(), src2_buf.data());
    if (err != 0) {
      std::cout << "found " << std::dec << err << " errors!" << std::endl;
      std::cout << "Test" << t << "-" << name << " FAILED!" << std::endl << std::flush;
      errors += err;
      if (stop_on_error)
        break;
    } else {
      std::cout << "Test" << t << "-" << name << " PASSED!" << std::endl << std::flush;
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  return errors;
}