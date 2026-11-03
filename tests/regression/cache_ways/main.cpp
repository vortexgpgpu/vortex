#include <iostream>
#include <unistd.h>
#include <vector>
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

// The last-level cache: a single-core build keeps it across launches.
#if VX_CFG_L3_ENABLED
static const char*    llc_name   = "L3";
static const uint32_t llc_size   = VX_CFG_L3_SIZE;
static const uint32_t llc_ways   = VX_CFG_L3_NUM_WAYS;
static const uint32_t llc_line   = VX_CFG_L3_LINE_SIZE;
static const uint32_t llc_sector = VX_CFG_L3_SECTOR_SIZE;
static const uint32_t llc_policy = VX_CFG_L3_REPL_POLICY;
#elif VX_CFG_L2_ENABLED
static const char*    llc_name   = "L2";
static const uint32_t llc_size   = VX_CFG_L2_SIZE;
static const uint32_t llc_ways   = VX_CFG_L2_NUM_WAYS;
static const uint32_t llc_line   = VX_CFG_L2_LINE_SIZE;
static const uint32_t llc_sector = VX_CFG_L2_SECTOR_SIZE;
static const uint32_t llc_policy = VX_CFG_L2_REPL_POLICY;
#else
static const char*    llc_name   = "dcache";
static const uint32_t llc_size   = VX_CFG_DCACHE_SIZE;
static const uint32_t llc_ways   = VX_CFG_DCACHE_NUM_WAYS;
static const uint32_t llc_line   = VX_CFG_L1_LINE_SIZE;
static const uint32_t llc_sector = VX_CFG_L1_LINE_SIZE;
static const uint32_t llc_policy = VX_CFG_DCACHE_REPL_POLICY;
#endif

// Lines a set-index apart: same set, same bank (without a bank hash).
static const uint32_t set_stride = llc_size / llc_ways;
static const uint32_t span_lines = set_stride / llc_line;
static const uint32_t sectors    = llc_line / llc_sector;

const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
std::vector<vx_buffer_h> buffers;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
bool counter_restarts = false;

static void show_usage() {
  std::cout << "Vortex Test." << std::endl;
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
    for (auto b : buffers) {
      vx_buffer_release(b);
    }
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static void launch(const kernel_arg_t& arg) {
  vx_event_h ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = 1;
  li.block_dim[0] = 1;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &ev));
  RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
}

static uint64_t mem_reads() {
  uint64_t value = 0;
  RT_CHECK(vx_device_mpm_query(device, VX_DCR_MPM_CLASS_MEM, VX_CSR_MPM_MEM_READS, 0, &value));
  return value;
}

// `lines` lines `stride` apart, the first at LLC line `first_line` of the set
// span, in a fresh buffer.
static kernel_arg_t lines_at(uint32_t first_line, uint32_t lines, uint32_t stride) {
  vx_buffer_h buffer = nullptr;
  uint64_t addr = 0;
  RT_CHECK(vx_buffer_create(device, (lines + 1) * stride + set_stride, VX_MEM_READ_WRITE, &buffer));
  buffers.push_back(buffer);
  RT_CHECK(vx_buffer_address(buffer, &addr));
  uint32_t base_line = (addr / llc_line) % span_lines;
  kernel_arg_t arg = {};
  arg.addr         = addr + (uint64_t)((first_line + span_lines - base_line) % span_lines) * llc_line;
  arg.lines        = lines;
  arg.line_stride  = stride;
  arg.sectors      = sectors;
  arg.sector_bytes = llc_sector;
  return arg;
}

// Memory reads of a second pass over lines the first pass left in the cache.
static int64_t second_pass_reads(const kernel_arg_t& arg) {
  launch(arg);
  uint64_t before = mem_reads();
  launch(arg);
  uint64_t after = mem_reads();
  return counter_restarts ? (int64_t)after : (int64_t)(after - before);
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << llc_name << ": " << llc_size << " bytes, " << llc_ways << " ways, " << llc_line << "-byte lines, "
            << sectors << " sectors per line, policy " << llc_policy << std::endl;

  // Away from the page-aligned buffers' first lines and the span's ends.
  const uint32_t target = (span_lines * 5 / 8 + 11) % span_lines;

  // Drivers differ on whether the memory read counter restarts at each
  // launch: one that reads nothing after one that reads a lot tells.
  launch(lines_at((target + span_lines / 4) % span_lines, 64, llc_line));
  uint64_t busy = mem_reads();
  kernel_arg_t none = {};
  launch(none);
  uint64_t idle = mem_reads();
  if (busy == 0) {
    std::cout << "error: memory read counter inactive (run with --perf)" << std::endl;
    std::cout << "FAILED!" << std::endl;
    cleanup();
    return 1;
  }
  counter_restarts = (idle < busy);

  // The same accesses spread over distinct sets: every line stays resident.
  int64_t control = second_pass_reads(lines_at((target + span_lines / 2) % span_lines, llc_ways, llc_line));
  // One set filled to its ways, every sector of each line read.
  int64_t fit = second_pass_reads(lines_at(target, llc_ways, set_stride));
  // One line more than the set holds.
  int64_t over = second_pass_reads(lines_at(target, llc_ways + 1, set_stride));

  std::cout << "second-pass memory reads: control " << control << ", " << llc_ways << " lines in one set " << fit
            << ", " << (llc_ways + 1) << " lines in one set " << over << std::endl;

  // A random victim may evict a resident line while another way is free.
  int errors = 0;
  const int64_t slack = sectors;
  if (llc_policy == 0) {
    std::cout << "random replacement: occupancy not checked" << std::endl;
  } else if (over - control <= slack) {
    // the set did not overflow: the lines do not share a set
    std::cout << "error: " << (llc_ways + 1) << " same-set lines all stayed resident" << std::endl;
    ++errors;
  } else if (fit - control > slack) {
    std::cout << "error: a set did not hold " << llc_ways << " lines" << std::endl;
    ++errors;
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
