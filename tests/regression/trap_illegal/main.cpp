// Illegal-instruction traps raised at decode: FP instructions, FP loads, FP
// CSR accesses and FP moves with mstatus.FS=Off, and reserved branch and
// SYSTEM encodings, each at the faulting pc; WFI runs without trapping. An FP
// write moves FS from Initial to Dirty and sets SD.
#include <vortex2.h>
#include "common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>

#define CHECK(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n", \
                     __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        std::exit(1); \
    } \
} while (0)

namespace {

constexpr int64_t CAUSE_ILLEGAL_INSTR = 2;
constexpr uint64_t MSTATUS_FS_SHIFT = 13;
constexpr uint64_t FS_INITIAL = 1, FS_DIRTY = 3;

const char* probe_names[] = {
  "fadd.s with FS=Off", "flw with FS=Off", "csrr fflags with FS=Off", "fmv.x.w with FS=Off",
  "branch funct3=2", "branch funct3=3", "reserved SYSTEM", "SYSTEM funct3=4", "wfi",
};

} // namespace

int main(int argc, char** argv) {
  const char* kernel_file = "kernel.vxbin";
  int c;
  while ((c = getopt(argc, argv, "k:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    default:
      std::cout << "Usage: [-k kernel] [-h]" << std::endl;
      return (c == 'h') ? 0 : -1;
    }
  }

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  uint64_t isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));
  uint32_t xlen = VX_ISA_ARCH(isa);

  vx_queue_h q = nullptr;
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  CHECK(vx_queue_create(dev, &qi, &q));
  vx_module_h mod = nullptr;
  vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  trap_out_t out{};
  vx_buffer_h dst = nullptr;
  kernel_arg_t arg{};
  CHECK(vx_buffer_create(dev, sizeof(out), VX_MEM_WRITE, &dst));
  CHECK(vx_buffer_address(dst, &arg.dst_addr));

  // One thread: the probes install their own mtvec, which is per core.
  vx_launch_info_t li{};
  li.struct_size = sizeof(li);
  li.kernel      = kern;
  li.args_host   = &arg;
  li.args_size   = sizeof(arg);
  li.ndim        = 1;
  li.grid_dim[0] = 1;
  li.block_dim[0]= 1;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, &out, dst, 0, sizeof(out), 1, &launch_ev, &read_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (int i = 0; i < NUM_PROBES; ++i) {
    const probe_t& p = out.probes[i];
    int64_t expected = (i == PROBE_WFI) ? -1 : CAUSE_ILLEGAL_INSTR;
    int64_t expected_offset = (i == PROBE_WFI) ? -1 : 0;
    if (p.cause != expected || p.epc_offset != expected_offset) {
      std::printf("*** %s: expected mcause %lld, mepc offset %lld; actual mcause %lld, mepc offset %lld\n",
                  probe_names[i], (long long)expected, (long long)expected_offset,
                  (long long)p.cause, (long long)p.epc_offset);
      ++errors;
    }
  }

  auto fs = [](uint64_t ms) { return (ms >> MSTATUS_FS_SHIFT) & 3; };
  auto sd = [xlen](uint64_t ms) { return (ms >> (xlen - 1)) & 1; };
  if (fs(out.mstatus_initial) != FS_INITIAL || sd(out.mstatus_initial) != 0) {
    std::printf("*** before an FP write: expected FS=%llu SD=0, actual FS=%llu SD=%llu (mstatus=0x%llx)\n",
                (unsigned long long)FS_INITIAL, (unsigned long long)fs(out.mstatus_initial),
                (unsigned long long)sd(out.mstatus_initial), (unsigned long long)out.mstatus_initial);
    ++errors;
  }
  if (fs(out.mstatus_dirty) != FS_DIRTY || sd(out.mstatus_dirty) != 1) {
    std::printf("*** after an FP write: expected FS=%llu SD=1, actual FS=%llu SD=%llu (mstatus=0x%llx)\n",
                (unsigned long long)FS_DIRTY, (unsigned long long)fs(out.mstatus_dirty),
                (unsigned long long)sd(out.mstatus_dirty), (unsigned long long)out.mstatus_dirty);
    ++errors;
  }

  vx_buffer_release(dst);
  vx_kernel_release(kern);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_release(dev);

  if (errors) {
    std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
