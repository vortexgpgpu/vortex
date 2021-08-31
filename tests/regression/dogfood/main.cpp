#include <iostream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <VX_config.h>
#include "testcases.h"
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

class TestMngr {
public:
  TestMngr() {
    this->add_test("iadd", new Test_IADD());
    this->add_test("imul", new Test_IMUL());
    this->add_test("idiv", new Test_IDIV());
    this->add_test("idiv-mul", new Test_IDIV_MUL());
  #ifdef EXT_F_ENABLE
    this->add_test("fadd", new Test_FADD());
    this->add_test("fsub", new Test_FSUB());
    this->add_test("fmul", new Test_FMUL());
    this->add_test("fmadd", new Test_FMADD());
    this->add_test("fmsub", new Test_FMSUB());
    this->add_test("fnmadd", new Test_FNMADD());
    this->add_test("fnmsub", new Test_FNMSUB());
    this->add_test("fnmadd-madd", new Test_FNMADD_MADD());
    this->add_test("fdiv", new Test_FDIV());
    this->add_test("fdiv2", new Test_FDIV2());
    this->add_test("fsqrt", new Test_FSQRT());
    this->add_test("ftoi", new Test_FTOI());
    this->add_test("ftou", new Test_FTOU());
    this->add_test("itof", new Test_ITOF());
    this->add_test("utof", new Test_UTOF());
  #endif
  }

  ~TestMngr() {
    for (size_t i = 0; i < _tests.size(); ++i) {
      delete _tests[i];
    }
  }

  const std::string& get_name(int testid) const {
    return _names.at(testid);
  }

  ITestCase* get_test(int testid) const {
    return _tests.at(testid);
  }

  void add_test(const char* name, ITestCase* test) {
    _names.push_back(name);
    _tests.push_back(test);
  }

  size_t size() const {
    return _tests.size();
  }  

private:
  std::vector<std::string> _names;
  std::vector<ITestCase*> _tests;
};

///////////////////////////////////////////////////////////////////////////////

TestMngr testMngr;
const char* kernel_file = "kernel.bin";
int count    = 0;
int testid_s = 0;
int testid_e = (testMngr.size() - 1);
bool stop_on_error = true;

vx_device_h device   = nullptr;
vx_buffer_h arg_buf  = nullptr;
vx_buffer_h src1_buf = nullptr;
vx_buffer_h src2_buf = nullptr;
vx_buffer_h dst_buf  = nullptr;

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-t:testid] [-s:testid] [-e:testid] [-k: kernel] [-n words] [-c] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:s:e:k:ch?")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
      break;
    case 't':
      testid_s = atoi(optarg);
      testid_e = atoi(optarg);
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
  if (arg_buf) {
    vx_buf_release(arg_buf);
  }
   if (src1_buf) {
    vx_buf_release(src1_buf);
  }
  if (src2_buf) {
    vx_buf_release(src2_buf);
  }
  if (dst_buf) {
    vx_buf_release(dst_buf);
  }
  if (device) {
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  int exitcode = 0;
  size_t value; 
  kernel_arg_t kernel_arg;
  
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
  RT_CHECK(vx_dev_open(&device));

  unsigned max_cores, max_warps, max_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_CORES, &max_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_WARPS, &max_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_THREADS, &max_threads));

  int num_tasks  = max_cores * max_warps * max_threads;
  int num_points = count * num_tasks;
  size_t buf_size = num_points * sizeof(uint32_t);
  
  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // upload program
  std::cout << "upload kernel" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;  

  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.src0_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.src1_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.dst_ptr = value;

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.task_size = count;

  std::cout << "dev_src0=" << std::hex << kernel_arg.src0_ptr << std::dec << std::endl;
  std::cout << "dev_src1=" << std::hex << kernel_arg.src1_ptr << std::dec << std::endl;
  std::cout << "dev_dst=" << std::hex << kernel_arg.dst_ptr << std::dec << std::endl;
  
  // allocate shared memory  
  std::cout << "allocate shared memory" << std::endl;
  RT_CHECK(vx_alloc_shared_mem(device, sizeof(kernel_arg_t), &arg_buf));
  RT_CHECK(vx_alloc_shared_mem(device, buf_size, &src1_buf));
  RT_CHECK(vx_alloc_shared_mem(device, buf_size, &src2_buf));
  RT_CHECK(vx_alloc_shared_mem(device, buf_size, &dst_buf));

  for (int t = testid_s; t <= testid_e; ++t) { 
    auto name = testMngr.get_name(t);
    auto test = testMngr.get_test(t);

    std::cout << "Test" << t << ": " << name << std::endl;

    // upload kernel argument
    std::cout << "upload kernel argument" << std::endl;
    kernel_arg.testid = t;
    memcpy((void*)vx_host_ptr(arg_buf), &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(arg_buf, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));

    // get test arguments
    std::cout << "get test arguments" << std::endl;
    test->setup(num_points, (void*)vx_host_ptr(src1_buf), (void*)vx_host_ptr(src2_buf));
    
    // upload source buffer0
    std::cout << "upload source buffer0" << std::endl;      
    RT_CHECK(vx_copy_to_dev(src1_buf, kernel_arg.src0_ptr, buf_size, 0));
    
    // upload source buffer1
    std::cout << "upload source buffer1" << std::endl;      
    RT_CHECK(vx_copy_to_dev(src2_buf, kernel_arg.src1_ptr, buf_size, 0));

    // clear destination buffer    
    std::cout << "clear destination buffer" << std::endl;     
    for (int i = 0; i < num_points; ++i) {
      ((uint32_t*)vx_host_ptr(dst_buf))[i] = 0xdeadbeef;
    }         
    RT_CHECK(vx_copy_to_dev(dst_buf, kernel_arg.dst_ptr, buf_size, 0));

    // start device
    std::cout << "start device" << std::endl;
    RT_CHECK(vx_start(device));

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_ready_wait(device, -1));

    // download destination buffer
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_copy_from_dev(dst_buf, kernel_arg.dst_ptr, buf_size, 0));

    // verify destination
    std::cout << "verify test result" << std::endl;
    int errors = test->verify(num_points, 
                              (void*)vx_host_ptr(dst_buf), 
                              (void*)vx_host_ptr(src1_buf), 
                              (void*)vx_host_ptr(src2_buf));
    if (errors != 0) {
      std::cout << "found " << std::dec << errors << " errors!" << std::endl;
      std::cout << "Test" << t << "-" << name << " FAILED!" << std::endl << std::flush;
      if (stop_on_error) {
        cleanup();
        exit(1);  
      }
      exitcode = 1;
    } else {
      std::cout << "Test" << t << "-" << name << " PASSED!" << std::endl << std::flush;
    }
  } 

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  return exitcode;
}