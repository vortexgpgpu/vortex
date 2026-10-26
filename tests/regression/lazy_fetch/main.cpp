#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";
uint32_t rounds = 16;
bool full_mode = false;

vx_device_h device = nullptr;
vx_buffer_h buf_buffer = nullptr;
vx_buffer_h err_buffer = nullptr;
vx_buffer_h set_buffer = nullptr;
std::vector<vx_buffer_h> full_buffers;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;

static void show_usage() {
  std::cout << "Vortex Test." << std::endl;
  std::cout << "Usage: [-k: kernel] [-n rounds] [-f: fully written lines read no memory] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:fh")) != -1) {
    switch (c) {
    case 'n':
      rounds = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'f':
      full_mode = true;
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
    if (buf_buffer) vx_buffer_release(buf_buffer);
    if (err_buffer) vx_buffer_release(err_buffer);
    if (set_buffer) vx_buffer_release(set_buffer);
    for (auto b : full_buffers) {
      vx_buffer_release(b);
    }
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static int launch(const kernel_arg_t& arg, uint32_t grid, uint32_t block) {
  vx_event_h ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = grid;
  li.block_dim[0] = block;
  int ret = vx_enqueue_launch(queue, &li, 0, nullptr, &ev);
  if (ret != 0) {
    return ret;
  }
  ret = vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
  vx_event_release(ev);
  return ret;
}

static int mem_reads(uint64_t* value) {
  return vx_device_mpm_query(device, VX_DCR_MPM_CLASS_MEM, VX_CSR_MPM_MEM_READS, 0, value);
}

static vx_buffer_h new_buffer(uint32_t size, uint64_t* addr) {
  vx_buffer_h buffer = nullptr;
  RT_CHECK(vx_buffer_create(device, size, VX_MEM_READ_WRITE, &buffer));
  full_buffers.push_back(buffer);
  RT_CHECK(vx_buffer_address(buffer, addr));
  return buffer;
}

static void read_buffer(void* dst, vx_buffer_h buffer, uint32_t size) {
  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, dst, buffer, 0, size, 0, nullptr, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
}

static int check_errors(uint32_t num_threads) {
  std::vector<uint32_t> h_err(num_threads);
  read_buffer(h_err.data(), err_buffer, num_threads * sizeof(uint32_t));
  int errors = 0;
  for (uint32_t t = 0; t < num_threads; ++t) {
    if (h_err[t] != 0) {
      if (errors < 16) {
        std::cout << "error: thread " << t << " read back 0x" << std::hex << h_err[t] << std::dec
                  << " wrong words of its lines" << std::endl;
      }
      ++errors;
    }
  }
  return errors;
}

static const char* full_name(const kernel_arg_t& arg) {
  switch (arg.full_mode) {
  case FULL_CHAIN: return arg.full_skip ? "chained, partially written" : "chained, fully written";
  case FULL_MIXED: return "mixed";
  default:         return arg.full_skip ? "partially written" : "fully written";
  }
}

// One phase-2 launch over a fresh pre-patterned buffer, after a launch that
// reads far more. Drivers differ on whether the memory read counter restarts
// at each launch; after the flood, a counter that restarted reads below its
// previous value. Returns the launch's memory reads.
static int64_t run_lines(kernel_arg_t arg, uint32_t grid, uint32_t block, uint32_t lines, int* errors) {
  const uint32_t flood_lines = 4096;
  uint32_t line_threads = (arg.full_mode == FULL_CHAIN) ? arg.num_blocks : arg.num_threads;
  uint32_t words = lines * line_threads * FULL_LINE_WORDS;
  vx_buffer_h buffer = new_buffer(words * sizeof(uint32_t), &arg.full_addr);
  std::vector<uint32_t> h_buf(words);
  for (uint32_t i = 0; i < words; ++i) {
    h_buf[i] = pattern_word(i);
  }
  RT_CHECK(vx_enqueue_write(queue, buffer, 0, h_buf.data(), words * sizeof(uint32_t), 0, nullptr, nullptr));

  kernel_arg_t flood = arg;
  flood.phase = 3;
  flood.full_lines = flood_lines / arg.num_threads;
  new_buffer(flood_lines * FLOOD_LINE_BYTES, &flood.full_addr);
  RT_CHECK(launch(flood, grid, block));
  *errors += check_errors(arg.num_threads);

  arg.phase = 2;
  arg.full_lines = lines;
  uint64_t before = 0, after = 0;
  RT_CHECK(mem_reads(&before));
  RT_CHECK(launch(arg, grid, block));
  RT_CHECK(mem_reads(&after));
  int64_t reads = (after < before) ? (int64_t)after : (int64_t)(after - before);
  *errors += check_errors(arg.num_threads);

  read_buffer(h_buf.data(), buffer, words * sizeof(uint32_t));
  int bad = 0;
  for (uint32_t i = 0; i < words; ++i) {
    uint32_t ref = full_expected(i, &arg);
    if (h_buf[i] != ref) {
      if (bad < 16) {
        std::cout << "error: " << full_name(arg) << " word " << i << std::hex << ": actual 0x" << h_buf[i]
                  << ", expected 0x" << ref << std::dec << std::endl;
      }
      ++bad;
    }
  }
  *errors += bad;
  std::cout << full_name(arg) << " lines, memory reads: " << reads << std::endl;
  return reads;
}

// Lines stored whole and then loaded must not be read from memory. Each pair
// of launches stores and loads the same lines, one leaving a word of each
// sector unwritten: every partial sector is read from memory once, a whole
// one never, and the rest of the two launches' traffic is alike.
static int run_full(kernel_arg_t arg, uint32_t grid, uint32_t block) {
  // few enough lines that a 16 KB cache keeps them resident
  const uint32_t lines = 2;
  int errors = 0;
  for (uint32_t mode : {FULL_SEPARATE, FULL_CHAIN}) {
    // a chained sector takes a whole block, which needs a word per thread
    if (mode == FULL_CHAIN && block < SECTOR_WORDS) {
      std::cout << "chained lines skipped: " << block << " threads per block" << std::endl;
      continue;
    }
    arg.full_mode = mode;
    uint32_t line_threads = (mode == FULL_CHAIN) ? arg.num_blocks : arg.num_threads;
    // a chained sector belongs to a block: give the blocks more lines
    uint32_t mode_lines = (mode == FULL_CHAIN) ? lines * 16 : lines;
    arg.full_skip = 1;
    int64_t partial = run_lines(arg, grid, block, mode_lines, &errors);
    arg.full_skip = 0;
    int64_t full = run_lines(arg, grid, block, mode_lines, &errors);
    if (partial == 0) {
      std::cout << "error: memory read counter inactive (run with --perf)" << std::endl;
      return errors + 1;
    }
    int64_t sectors = (int64_t)mode_lines * line_threads * (FULL_LINE_WORDS / SECTOR_WORDS);
    int64_t saved = partial - full;
    // A sector evicted before its load is read back from memory either way.
    int64_t slack = std::max<int64_t>(2, sectors / 16);
    std::cout << full_name(arg) << " sectors: " << sectors << ", memory reads saved: " << saved << std::endl;
    if (saved < sectors - slack || saved > sectors + slack) {
      std::cout << "error: fully written sectors were read from memory" << std::endl;
      ++errors;
    }
  }
  // Loads of whole sectors loop back while partial ones read memory.
  arg.full_mode = FULL_MIXED;
  run_lines(arg, grid, block, lines, &errors);
  return errors;
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  if (rounds == 0) {
    rounds = 1;
  }

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores, num_warps, num_threads, isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));

  // One block per core, holding all of its warps.
  uint32_t block = num_warps * num_threads;
  uint32_t grid  = num_cores;
  uint32_t total_threads = grid * block;
  uint32_t num_regions = rounds * total_threads * NUM_CASES;
  uint32_t num_words = num_regions * REGION_WORDS;
  uint32_t buf_size = num_words * sizeof(uint32_t);
  uint32_t err_size = 2 * total_threads * sizeof(uint32_t);
  uint32_t set_words = rounds * total_threads * SET_STRIDE_WORDS;
  uint32_t set_size = set_words * sizeof(uint32_t);

  kernel_arg_t arg = {};
  arg.rounds      = rounds;
  arg.num_threads = total_threads;
  arg.warp_size   = num_threads;
  arg.cross_warp  = (num_warps % 2) == 0;
  arg.amo         = (isa_flags & VX_ISA_STD_A) != 0;
  arg.num_blocks  = grid;

  std::cout << "threads: " << total_threads << ", amo: " << arg.amo << ", regions: " << num_regions
            << ", buffer size: " << buf_size << " bytes" << std::endl;

  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ_WRITE, &buf_buffer));
  RT_CHECK(vx_buffer_address(buf_buffer, &arg.buf_addr));
  RT_CHECK(vx_buffer_create(device, err_size, VX_MEM_READ_WRITE, &err_buffer));
  RT_CHECK(vx_buffer_address(err_buffer, &arg.err_addr));
  RT_CHECK(vx_buffer_create(device, set_size, VX_MEM_READ_WRITE, &set_buffer));
  RT_CHECK(vx_buffer_address(set_buffer, &arg.set_addr));

  std::vector<uint8_t> h_buf(buf_size);
  for (uint32_t i = 0; i < buf_size; ++i) {
    h_buf[i] = pattern_byte(i);
  }
  std::vector<uint32_t> h_err(2 * total_threads, 0xdeadbeef);
  std::vector<uint32_t> h_set(set_words);
  for (uint32_t i = 0; i < set_words; ++i) {
    h_set[i] = pattern_word(i);
  }

  std::cout << "upload buffers" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, buf_buffer, 0, h_buf.data(), buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, err_buffer, 0, h_err.data(), err_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, set_buffer, 0, h_set.data(), set_size, 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  if (full_mode) {
    int errors = run_full(arg, grid, block);
    cleanup();
    if (errors != 0) {
      std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
      std::cout << "FAILED!" << std::endl;
      return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
  }

  std::cout << "launch kernel (exercise)" << std::endl;
  arg.phase = 0;
  RT_CHECK(launch(arg, grid, block));

  std::cout << "launch kernel (re-check)" << std::endl;
  arg.phase  = 1;
  arg.rotate = (num_cores > 1) ? block : num_threads;
  RT_CHECK(launch(arg, grid, block));

  std::cout << "download buffers" << std::endl;
  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_buf.data(), buf_buffer, 0, buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_read(queue, h_set.data(), set_buffer, 0, set_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_read(queue, h_err.data(), err_buffer, 0, err_size, 0, nullptr, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);

  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t t = 0; t < 2 * total_threads; ++t) {
    if (h_err[t] != 0) {
      if (errors < 16) {
        std::cout << "error: thread " << (t % total_threads) << (t < total_threads ? " (exercise)" : " (re-check)")
                  << " reported 0x" << std::hex << h_err[t] << std::dec << " mismatches" << std::endl;
      }
      ++errors;
    }
  }
  auto words = reinterpret_cast<const uint32_t*>(h_buf.data());
  for (uint32_t region = 0; region < num_regions; ++region) {
    for (uint32_t w = 0; w < REGION_WORDS; ++w) {
      uint32_t ref = expected_word(region, w, &arg);
      uint32_t cur = words[region * REGION_WORDS + w];
      if (cur != ref) {
        if (errors < 16) {
          std::cout << "error: region " << region << " (case " << (region % NUM_CASES) << ") word " << w
                    << std::hex << ": actual 0x" << cur << ", expected 0x" << ref << std::dec << std::endl;
        }
        ++errors;
      }
    }
  }

  for (uint32_t i = 0; i < set_words; ++i) {
    uint32_t ref = expected_set_word(i);
    if (h_set[i] != ref) {
      if (errors < 16) {
        std::cout << "error: same-set word " << i << std::hex << ": actual 0x" << h_set[i]
                  << ", expected 0x" << ref << std::dec << std::endl;
      }
      ++errors;
    }
  }

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
