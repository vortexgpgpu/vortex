#include <vortex2.h>
#include <rvfloats.h>
#include "common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
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

const char* kernel_file = "kernel.vxbin";
uint32_t    num_random  = 4096;
uint32_t    frm_dyn     = 3;
uint32_t    phases      = 0x3f;
uint32_t    verbose     = 0;
uint32_t    num_flag_in = 160;

vx_device_h dev = nullptr;
vx_queue_h  q = nullptr;
vx_kernel_h kern = nullptr;
uint32_t    xlen = 32;
int         errors = 0;

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:m:p:v:f:k:h")) != -1) {
    switch (c) {
    case 'n': num_random  = std::atoi(optarg); break;
    case 'p': phases      = std::strtoul(optarg, nullptr, 0); break;
    case 'v': verbose     = std::atoi(optarg); break;
    case 'f': num_flag_in = std::atoi(optarg); break;
    case 'm': frm_dyn     = std::atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    default:
      std::cout << "Usage: [-k kernel] [-n random cases] [-m dynamic frm] [-p phase mask] [-v dump count] [-f flag inputs] [-h]" << std::endl;
      std::exit(c == 'h' ? 0 : -1);
    }
  }
}

struct Buf {
  vx_buffer_h h = nullptr;
  uint64_t addr = 0;
  uint64_t size = 0;
  Buf(uint64_t sz) : size(sz ? sz : 8) {
    CHECK(vx_buffer_create(dev, size, VX_MEM_READ_WRITE, &h));
    CHECK(vx_buffer_address(h, &addr));
  }
  ~Buf() { vx_buffer_release(h); }
  void write(const void* p) { CHECK(vx_enqueue_write(q, h, 0, p, size, 0, nullptr, nullptr)); }
};

void run(uint32_t mode, uint32_t count, Buf& src, Buf& dst, Buf* aux, void* out, uint64_t out_size) {
  kernel_arg_t arg{};
  arg.mode = mode;
  arg.count = count;
  arg.frm = frm_dyn;
  arg.src_addr = src.addr;
  arg.dst_addr = dst.addr;
  arg.aux_addr = aux ? aux->addr : 0;
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
  CHECK(vx_enqueue_read(q, out, dst.h, 0, out_size, 1, &launch_ev, &read_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
}

void read_aux(Buf& aux, void* out, uint64_t size) {
  vx_event_h ev = nullptr;
  CHECK(vx_enqueue_read(q, out, aux.h, 0, size, 0, nullptr, &ev));
  CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
}

void report(const char* what, uint64_t input, uint32_t rm, uint64_t expected, uint64_t actual) {
  if (errors < 32) {
    std::printf("*** %s in=0x%llx rm=%u expected=0x%llx actual=0x%llx\n", what,
                (unsigned long long)input, rm, (unsigned long long)expected, (unsigned long long)actual);
  }
  ++errors;
}

uint32_t rm_of(uint32_t k) { return (k < 5) ? k : frm_dyn; }

int64_t sext16(uint64_t v) { return (int16_t)(uint16_t)v; }

std::vector<uint32_t> f32_inputs(std::mt19937& rng) {
  std::vector<uint32_t> v = {
    0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000,
    0x7f800001, 0xff800001, 0x7fa00000, 0x7fc12345, 0x7f7fffff, 0xff7fffff,
    0x00000001, 0x80000001, 0x007fffff, 0x00800000,
    0x477fe000, 0x477fefff, 0x477ff000, 0x477ff001, 0x477fffff, 0x47800000,
    0xc77fe000, 0xc77ff000, 0xc7800000,
    0x38800000, 0x387fffff, 0x387fe000, 0x387ff000, 0x33800000, 0x33000000,
    0x33000001, 0x32ffffff, 0x33400000, 0x33c00000, 0x2f800000, 0xb3000000,
    0xb3000001, 0x3f800000, 0x3f801000, 0x3f803000, 0x3f801001, 0x3f800fff,
    0xbf801000, 0xbf803000, 0x40490fdb, 0x3eaaaaab,
  };
  const uint32_t low[] = {0, 1, 0xfff, 0x1000, 0x1001, 0x1fff, 0x2000, 0x3000, 0x7fffff, 0x7fefff};
  for (int e = -27; e <= 17; ++e) {
    for (uint32_t l : low) {
      for (int s = 0; s < 2; ++s) {
        uint32_t hi = rng() & 0x7fe000;
        v.push_back((uint32_t(s) << 31) | (uint32_t(e + 127) << 23) | ((hi | l) & 0x7fffff));
      }
    }
  }
  for (uint32_t i = 0; i < num_random; ++i) {
    uint32_t e = 127 - 30 + (rng() % 50);
    v.push_back((rng() & 0x807fffff) | (e << 23));
  }
  for (uint32_t i = 0; i < num_random / 4; ++i) {
    v.push_back(rng());
  }
  return v;
}

std::vector<uint64_t> f64_inputs(std::mt19937& rng) {
  std::vector<uint64_t> v = {
    0x0000000000000000ull, 0x8000000000000000ull, 0x7ff0000000000000ull, 0xfff0000000000000ull,
    0x7ff8000000000000ull, 0x7ff0000000000001ull, 0x7ff4000000000000ull, 0x7fefffffffffffffull,
    0x0000000000000001ull, 0x40effc0000000000ull, 0x40effdffffffffffull, 0x40effe0000000000ull,
    0x40effe0000000001ull, 0x40f0000000000000ull, 0x3f10000000000000ull, 0x3e70000000000000ull,
    0x3e60000000000000ull, 0x3e60000000000001ull, 0x3e5fffffffffffffull, 0x3ff0000000000000ull,
    0x3ff0020000000000ull, 0x3ff0020000000001ull, 0x3ff001ffffffffffull, 0xbff0060000000000ull,
  };
  for (int e = -27; e <= 17; ++e) {
    for (int k = 0; k < 8; ++k) {
      uint64_t man = ((uint64_t(rng()) << 32) | rng()) & 0xfffffffffffffull;
      if (k & 1) {
        man = (man & ~0x3ffffffffffull) | 0x20000000000ull;
      }
      if (k & 2) {
        man &= ~0x3ffffffffffull;
      }
      v.push_back((uint64_t(k & 4 ? 1 : 0) << 63) | (uint64_t(e + 1023) << 52) | man);
    }
  }
  for (uint32_t i = 0; i < num_random; ++i) {
    uint64_t e = 1023 - 30 + (rng() % 50);
    uint64_t man = ((uint64_t(rng()) << 32) | rng()) & 0x800fffffffffffffull;
    v.push_back(man | (e << 52));
  }
  return v;
}

void test_narrow_s(const std::vector<uint32_t>& in) {
  uint32_t n = in.size();
  Buf src(n * 4), dst(n * ZF_NUM_RM * 2), aux(n * 8);
  src.write(in.data());
  std::vector<uint16_t> out(n * ZF_NUM_RM);
  std::vector<int64_t> mvx(n);
  run(ZF_MODE_NARROW_S, n, src, dst, &aux, out.data(), dst.size);
  read_aux(aux, mvx.data(), aux.size);
  for (uint32_t i = 0; i < n && i < verbose; ++i) {
    std::printf("in=0x%08x", in[i]);
    for (uint32_t k = 0; k < ZF_NUM_RM; ++k) {
      std::printf(" %04x/%04x", out[i * ZF_NUM_RM + k], rv_ftoh_s(in[i], rm_of(k), nullptr));
    }
    std::printf("\n");
  }
  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t k = 0; k < ZF_NUM_RM; ++k) {
      uint16_t ref = rv_ftoh_s(in[i], rm_of(k), nullptr);
      if (out[i * ZF_NUM_RM + k] != ref) {
        report("fcvt.h.s", in[i], rm_of(k), ref, out[i * ZF_NUM_RM + k]);
      }
    }
    int64_t ref = sext16(rv_ftoh_s(in[i], 0, nullptr));
    if (mvx[i] != ref) {
      report("fmv.x.h", in[i], 0, ref, mvx[i]);
    }
  }
  std::printf("fcvt.h.s: %u inputs x %u rounding modes\n", n, ZF_NUM_RM);
}

void test_widen_s() {
  uint32_t n = 65536;
  std::vector<uint16_t> in(n);
  for (uint32_t i = 0; i < n; ++i) {
    in[i] = i;
  }
  Buf src(n * 2), dst(n * 4);
  src.write(in.data());
  std::vector<uint32_t> out(n);
  run(ZF_MODE_WIDEN_S, n, src, dst, nullptr, out.data(), dst.size);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t ref = rv_htof_s(in[i], 0, nullptr);
    if (out[i] != ref) {
      report("fcvt.s.h", in[i], 0, ref, out[i]);
    }
  }
  std::printf("fcvt.s.h: all %u half inputs\n", n);
}

void test_narrow_d(const std::vector<uint64_t>& in) {
  uint32_t n = in.size();
  Buf src(n * 8), dst(n * ZF_NUM_RM * 2);
  src.write(in.data());
  std::vector<uint16_t> out(n * ZF_NUM_RM);
  run(ZF_MODE_NARROW_D, n, src, dst, nullptr, out.data(), dst.size);
  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t k = 0; k < ZF_NUM_RM; ++k) {
      uint16_t ref = rv_dtoh(in[i], rm_of(k), nullptr);
      if (out[i * ZF_NUM_RM + k] != ref) {
        report("fcvt.h.d", in[i], rm_of(k), ref, out[i * ZF_NUM_RM + k]);
      }
    }
  }
  std::printf("fcvt.h.d: %u inputs x %u rounding modes\n", n, ZF_NUM_RM);
}

void test_widen_d() {
  uint32_t n = 65536;
  std::vector<uint16_t> in(n);
  for (uint32_t i = 0; i < n; ++i) {
    in[i] = i;
  }
  Buf src(n * 2), dst(n * 8);
  src.write(in.data());
  std::vector<uint64_t> out(n);
  run(ZF_MODE_WIDEN_D, n, src, dst, nullptr, out.data(), dst.size);
  for (uint32_t i = 0; i < n; ++i) {
    uint64_t ref = rv_htod(in[i], 0, nullptr);
    if (out[i] != ref) {
      report("fcvt.d.h", in[i], 0, ref, out[i]);
    }
  }
  std::printf("fcvt.d.h: all %u half inputs\n", n);
}

void test_flags(const std::vector<uint32_t>& f32, const std::vector<uint64_t>& f64) {
  std::vector<zf_flag_case_t> cases;
  for (size_t i = 0; i < num_flag_in && i < f32.size(); ++i) {
    for (uint32_t rm = 0; rm < 5; ++rm) {
      cases.push_back({ZF_OP_H_S, rm, f32[i]});
    }
  }
  const uint16_t halves[] = {0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x7c00,
                             0xfc00, 0x7e00, 0x7c01, 0x7d00, 0xfe00, 0xfd55, 0x7fff, 0x1234};
  for (uint16_t h : halves) {
    cases.push_back({ZF_OP_S_H, 0, h});
  }
  if (xlen == 64) {
    for (size_t i = 0; i < 64 && i < f64.size(); ++i) {
      for (uint32_t rm = 0; rm < 5; ++rm) {
        cases.push_back({ZF_OP_H_D, rm, f64[i]});
      }
    }
    for (uint16_t h : halves) {
      cases.push_back({ZF_OP_D_H, 0, h});
    }
  }
  uint32_t n = cases.size();
  Buf src(n * sizeof(zf_flag_case_t)), dst(n * sizeof(zf_flag_out_t));
  src.write(cases.data());
  std::vector<zf_flag_out_t> out(n);
  run(ZF_MODE_FLAGS, n, src, dst, nullptr, out.data(), dst.size);
  static const char* names[] = {"fcvt.h.s", "fcvt.s.h", "fcvt.h.d", "fcvt.d.h"};
  for (uint32_t i = 0; i < n; ++i) {
    const auto& c = cases[i];
    uint32_t fl = 0;
    uint64_t ref = 0;
    switch (c.op) {
    case ZF_OP_H_S: ref = rv_ftoh_s(uint32_t(c.bits), c.rm, &fl); break;
    case ZF_OP_S_H: ref = rv_htof_s(uint16_t(c.bits), c.rm, &fl); break;
    case ZF_OP_H_D: ref = rv_dtoh(c.bits, c.rm, &fl); break;
    case ZF_OP_D_H: ref = rv_htod(uint16_t(c.bits), c.rm, &fl); break;
    }
    if (out[i].result != ref) {
      report(names[c.op], c.bits, c.rm, ref, out[i].result);
    }
    if (out[i].fflags != fl) {
      char what[32];
      std::snprintf(what, sizeof(what), "%s fflags", names[c.op]);
      report(what, c.bits, c.rm, fl, out[i].fflags);
    }
  }
  std::printf("fflags: %u single-lane cases\n", n);
}

void test_box(std::mt19937& rng) {
  uint32_t n = 1024;
  std::vector<uint16_t> h(n);
  std::vector<uint32_t> g(n);
  for (uint32_t i = 0; i < n; ++i) {
    h[i] = rng();
    g[i] = rng();
    if (i % 4 == 0) {
      g[i] |= 0xffff0000;
    }
    if (i % 8 == 1) {
      g[i] &= 0x0000ffff;
    }
  }
  Buf src(n * 2), aux(n * 4), dst(n * ZF_BOX_OUTS * 8);
  src.write(h.data());
  aux.write(g.data());
  std::vector<uint64_t> out(n * ZF_BOX_OUTS);
  run(ZF_MODE_BOX, n, src, dst, &aux, out.data(), dst.size);
  uint64_t box = (xlen == 64) ? 0xffffffffffff0000ull : 0xffff0000ull;
  for (uint32_t i = 0; i < n; ++i) {
    const uint64_t* o = &out[i * ZF_BOX_OUTS];
    uint64_t r0 = box | h[i];
    uint64_t r1 = box | (g[i] & 0xffff);
    uint64_t r2 = (uint64_t)sext16(h[i]);
    uint64_t r3 = ((g[i] >> 16) == 0xffff) ? rv_htof_s(uint16_t(g[i]), 0, nullptr) : 0x7fc00000u;
    uint64_t r4 = (uint64_t)sext16(g[i]);
    if (o[0] != r0) {
      report("flh box", h[i], 0, r0, o[0]);
    }
    if (o[1] != r1) {
      report("fmv.h.x box", g[i], 0, r1, o[1]);
    }
    if (o[2] != r2) {
      report("flh+fmv.x.h", h[i], 0, r2, o[2]);
    }
    if (o[3] != r3) {
      report("unboxed fcvt.s.h", g[i], 0, r3, o[3]);
    }
    if (o[4] != r4) {
      report("fmv.x.h unboxed", g[i], 0, r4, o[4]);
    }
  }
  std::printf("boxing: %u cases\n", n);
}

} // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);
  std::mt19937 rng(42);

  CHECK(vx_device_open(0, &dev));
  uint64_t isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));
  if (!(isa & VX_ISA_EXT_ZFHMIN)) {
    std::cout << "Error: device does not support Zfhmin (VX_CFG_EXT_ZFHMIN_ENABLE)" << std::endl;
    vx_device_release(dev);
    return 1;
  }
  xlen = VX_ISA_ARCH(isa);
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  CHECK(vx_queue_create(dev, &qi, &q));
  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  std::printf("zfhmin: xlen=%u dynamic frm=%u\n", xlen, frm_dyn);
  auto f32 = f32_inputs(rng);
  auto f64 = f64_inputs(rng);
  if (phases & 0x01) {
    test_narrow_s(f32);
  }
  if (phases & 0x02) {
    test_widen_s();
  }
  if (xlen == 64) {
    if (phases & 0x04) {
      test_narrow_d(f64);
    }
    if (phases & 0x08) {
      test_widen_d();
    }
  }
  if (phases & 0x10) {
    test_flags(f32, f64);
  }
  if (phases & 0x20) {
    test_box(rng);
  }

  vx_kernel_release(kern);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);

  if (errors) {
    std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
