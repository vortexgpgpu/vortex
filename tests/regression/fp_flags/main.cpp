// Bit-exact results and fflags of sign-injection, min/max and compare on NaN
// operands, checked against values taken from the RISC-V F/D specification
// (not softfloat, which the simulators share).
#include <vortex2.h>
#include "common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n", \
                     __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        std::exit(1); \
    } \
} while (0)

namespace {

constexpr uint32_t NV = 0x10;

constexpr uint64_t S_QNAN = 0x7fc00000, S_SNAN = 0x7fa00000, S_NSNAN = 0xffa00000;
constexpr uint64_t S_ONE = 0x3f800000, S_NONE = 0xbf800000, S_TWO = 0x40000000;

constexpr uint64_t D_QNAN = 0x7ff8000000000000, D_SNAN = 0x7ff4000000000000, D_NSNAN = 0xfff4000000000000;
constexpr uint64_t D_ONE = 0x3ff0000000000000, D_NONE = 0xbff0000000000000, D_TWO = 0x4000000000000000;

const char* op_names[] = {"fsgnj", "fsgnjn", "fsgnjx", "fmin", "fmax", "feq", "flt", "fle"};

struct Case {
  uint32_t op;
  uint64_t a, b;
  uint64_t result;
  uint32_t fflags;
  uint32_t raw = 0;
};

// One table per precision: sign injection never raises flags; fmin/fmax
// with two NaN operands return the canonical NaN and raise NV if either one is
// signaling. The compare rows check that NV still comes from compares.
std::vector<Case> cases(uint64_t qnan, uint64_t snan, uint64_t nsnan,
                        uint64_t one, uint64_t none, uint64_t two, uint32_t d) {
  return {
    {FF_OP_FSGNJ  + d, snan,  one,   snan,  0},
    {FF_OP_FSGNJN + d, snan,  one,   nsnan, 0},
    {FF_OP_FSGNJX + d, snan,  none,  nsnan, 0},
    {FF_OP_FSGNJ  + d, one,   nsnan, none,  0},
    {FF_OP_FSGNJN + d, snan,  snan,  nsnan, 0},
    {FF_OP_FSGNJX + d, qnan,  one,   qnan,  0},
    {FF_OP_FMIN   + d, snan,  snan,  qnan,  NV},
    {FF_OP_FMAX   + d, qnan,  snan,  qnan,  NV},
    {FF_OP_FMIN   + d, nsnan, qnan,  qnan,  NV},
    {FF_OP_FMAX   + d, snan,  qnan,  qnan,  NV},
    {FF_OP_FMIN   + d, qnan,  qnan,  qnan,  0},
    {FF_OP_FMAX   + d, qnan,  qnan,  qnan,  0},
    {FF_OP_FMIN   + d, snan,  one,   one,   NV},
    {FF_OP_FMAX   + d, qnan,  two,   two,   0},
    {FF_OP_FEQ    + d, snan,  one,   0,     NV},
    {FF_OP_FEQ    + d, qnan,  one,   0,     0},
    {FF_OP_FLT    + d, qnan,  one,   0,     NV},
    {FF_OP_FLE    + d, one,   two,   1,     0},
  };
}

// Single-precision ops on raw 64-bit registers: an operand whose upper half is
// not all ones reads as the canonical NaN, and every result is NaN-boxed.
constexpr uint64_t BOX = 0xffffffff00000000;
constexpr uint64_t R_UNBOXED_ONE = 0x000000003f800000;
constexpr uint64_t R_CANON = BOX | 0x7fc00000, R_NCANON = BOX | 0xffc00000;
constexpr uint64_t R_ONE = BOX | S_ONE, R_NONE = BOX | S_NONE, R_TWO = BOX | S_TWO;

std::vector<Case> raw_cases() {
  return {
    {FF_OP_FSGNJ,  R_UNBOXED_ONE, R_ONE,         R_CANON,  0,  1},
    {FF_OP_FSGNJN, R_UNBOXED_ONE, R_ONE,         R_NCANON, 0,  1},
    {FF_OP_FSGNJ,  R_ONE,         R_UNBOXED_ONE, R_ONE,    0,  1},
    {FF_OP_FSGNJN, R_ONE,         R_UNBOXED_ONE, R_NONE,   0,  1},
    {FF_OP_FSGNJ,  R_ONE,         R_TWO,         R_ONE,    0,  1},
    {FF_OP_FMIN,   R_UNBOXED_ONE, R_TWO,         R_TWO,    0,  1},
    {FF_OP_FMAX,   R_UNBOXED_ONE, R_UNBOXED_ONE, R_CANON,  0,  1},
    {FF_OP_FMIN,   R_ONE,         R_TWO,         R_ONE,    0,  1},
    {FF_OP_FEQ,    R_UNBOXED_ONE, R_ONE,         0,        0,  1},
    {FF_OP_FLT,    R_UNBOXED_ONE, R_ONE,         0,        NV, 1},
  };
}

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

  auto tests = cases(S_QNAN, S_SNAN, S_NSNAN, S_ONE, S_NONE, S_TWO, 0);
  if (xlen == 64) {
    auto dt = cases(D_QNAN, D_SNAN, D_NSNAN, D_ONE, D_NONE, D_TWO, FF_NUM_OPS);
    tests.insert(tests.end(), dt.begin(), dt.end());
    auto rt = raw_cases();
    tests.insert(tests.end(), rt.begin(), rt.end());
  }
  uint32_t count = tests.size();

  std::vector<ff_case_t> in(count);
  for (uint32_t i = 0; i < count; ++i) {
    in[i] = {tests[i].op, tests[i].raw, tests[i].a, tests[i].b};
  }
  std::vector<ff_out_t> out(count);

  vx_queue_h q = nullptr;
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  CHECK(vx_queue_create(dev, &qi, &q));
  vx_module_h mod = nullptr;
  vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  uint64_t in_size = count * sizeof(ff_case_t);
  uint64_t out_size = count * sizeof(ff_out_t);
  vx_buffer_h src = nullptr, dst = nullptr;
  kernel_arg_t arg{};
  arg.count = count;
  CHECK(vx_buffer_create(dev, in_size, VX_MEM_READ, &src));
  CHECK(vx_buffer_create(dev, out_size, VX_MEM_WRITE, &dst));
  CHECK(vx_buffer_address(src, &arg.src_addr));
  CHECK(vx_buffer_address(dst, &arg.dst_addr));
  CHECK(vx_enqueue_write(q, src, 0, in.data(), in_size, 0, nullptr, nullptr));

  uint32_t grid[1], block[1];
  CHECK(vx_device_max_occupancy_grid(dev, 1, &count, grid, block));
  vx_launch_info_t li{};
  li.struct_size = sizeof(li);
  li.kernel      = kern;
  li.args_host   = &arg;
  li.args_size   = sizeof(arg);
  li.ndim        = 1;
  li.grid_dim[0] = grid[0];
  li.block_dim[0]= block[0];
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, out.data(), dst, 0, out_size, 1, &launch_ev, &read_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const Case& t = tests[i];
    bool is_d = t.op >= FF_NUM_OPS;
    if (out[i].result != t.result || out[i].fflags != t.fflags) {
      std::printf("*** %s.%c a=0x%llx b=0x%llx: expected 0x%llx/fflags 0x%x, actual 0x%llx/fflags 0x%x\n",
                  op_names[t.op % FF_NUM_OPS], t.raw ? 'r' : (is_d ? 'd' : 's'),
                  (unsigned long long)t.a, (unsigned long long)t.b,
                  (unsigned long long)t.result, t.fflags,
                  (unsigned long long)out[i].result, out[i].fflags);
      ++errors;
    }
  }
  std::printf("fp_flags: xlen=%u, %u cases\n", xlen, count);

  vx_buffer_release(src);
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
