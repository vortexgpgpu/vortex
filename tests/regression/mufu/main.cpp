#include <iostream>
#include <unistd.h>
#include <string.h>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <vortex2.h>
#include <mufu.h>
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

static const char* kOpNames[OP_COUNT] = {
  "rcp", "rsqrt", "ex2", "lg2", "sinpi", "cospi", "tanh",
  "expf", "logf", "sinf", "cosf", "sqrtf", "sigmoid", "silu", "gelu"
};

static const uint32_t kSpecials[] = {
  0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00800000, 0x80800000,
  0x3f800000, 0xbf800000, 0x3f000000, 0x40000000, 0x40800000, 0x40490fdb,
  0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x7fa00000,
  0x42fe0000, 0x43000000, 0xc2fc0000, 0xc3000000, 0x4b000001, 0x3f7fffff,
  0x3f800001, 0x39800000, 0x41000000, 0x40ffffff,
};

const char* kernel_file = "kernel.vxbin";
uint32_t num_points = 4096;
uint32_t repeat = 1;
int selected_op = -1;

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;

static void show_usage() {
  std::cout << "Vortex MUFU Test." << std::endl;
  std::cout << "Usage: [-k: kernel] [-n points] [-r repeat] [-t op] [-h: help]" << std::endl;
  std::cout << "  ops:";
  for (auto name : kOpNames)
    std::cout << " " << name;
  std::cout << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:r:t:k:h")) != -1) {
    switch (c) {
    case 'n':
      num_points = atoi(optarg);
      break;
    case 'r':
      repeat = std::max(1, atoi(optarg));
      break;
    case 't':
      for (int i = 0; i < OP_COUNT; ++i) {
        if (strcmp(optarg, kOpNames[i]) == 0)
          selected_op = i;
      }
      if (selected_op < 0) {
        show_usage();
        exit(-1);
      }
      break;
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
    if (src_buffer) vx_buffer_release(src_buffer);
    if (dst_buffer) vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

///////////////////////////////////////////////////////////////////////////////

static float as_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t as_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static float rand_range(std::mt19937& rng, float lo, float hi) {
  return std::uniform_real_distribution<float>(lo, hi)(rng);
}

// Log-uniform magnitude in [2^lo, 2^hi], random sign when `sign`.
static float rand_log(std::mt19937& rng, float lo, float hi, bool sign) {
  float m = std::exp2(rand_range(rng, lo, hi));
  return (sign && (rng() & 1)) ? -m : m;
}

static float gen_input(int op, std::mt19937& rng) {
  // A quarter of raw-op inputs are arbitrary bit patterns.
  if (op <= OP_TANH && (rng() & 3) == 0)
    return as_float(uint32_t(rng()));
  switch (op) {
  case OP_RCP:     return rand_log(rng, -125, 125, true);
  case OP_RSQRT:   return rand_log(rng, -125, 125, false);
  case OP_EX2:     return rand_range(rng, -130.0f, 130.0f);
  case OP_LG2:     return (rng() & 1) ? rand_range(rng, 0.5f, 2.0f) : rand_log(rng, -125, 125, false);
  case OP_SINPI:
  case OP_COSPI:   return (rng() & 1) ? rand_range(rng, -2.0f, 2.0f) : rand_range(rng, -4096.0f, 4096.0f);
  case OP_TANH:    return (rng() & 1) ? rand_range(rng, -10.0f, 10.0f) : rand_log(rng, -20, 3, true);
  case OP_EXPF:    return rand_range(rng, -80.0f, 80.0f);
  case OP_LOGF:    return rand_log(rng, -120, 120, false);
  case OP_SINF:
  case OP_COSF:    return rand_range(rng, -100.0f, 100.0f);
  case OP_SQRTF:   return rand_log(rng, -120, 120, false);
  case OP_SIGMOID:
  case OP_SILU:
  case OP_GELU:    return rand_range(rng, -20.0f, 20.0f);
  default:         return 0.0f;
  }
}

// Raw-op accuracy against libm, with the bounds the MUFU spec guarantees.
static bool raw_accurate(int op, float x, float y) {
  double r;
  switch (op) {
  case OP_RCP:   r = 1.0 / x; break;
  case OP_RSQRT: r = 1.0 / std::sqrt((double)x); break;
  case OP_EX2:   r = std::exp2((double)x); break;
  case OP_LG2:   r = std::log2((double)x); break;
  case OP_SINPI: r = std::sin(M_PI * std::fmod((double)x, 2.0)); break;
  case OP_COSPI: r = std::cos(M_PI * std::fmod((double)x, 2.0)); break;
  case OP_TANH:  r = std::tanh((double)x); break;
  default:       return true;
  }
  // Out of scope for the bound: specials, and results the FTZ/overflow rules decide.
  if (!std::isnormal(x) || std::isnan(r) || std::isinf(r)
   || std::fabs(r) >= 3.4028234663852886e38 || std::fabs(r) < 1.1754943508222875e-38)
    return true;
  double err = std::fabs((double)y - r);
  int ex;
  std::frexp(r, &ex);
  double ulp = std::ldexp(1.0, ex - 24);
  switch (op) {
  case OP_LG2:   return (x >= 0.5f && x <= 2.0f) ? err <= 0x1p-22 : err <= 2 * ulp;
  case OP_SINPI:
  case OP_COSPI: return err <= 0x1p-21;
  case OP_TANH:  return err <= 0x1p-11 * std::fabs(r);
  default:       return err <= 2 * ulp;
  }
}

// Composite reference and tolerance; the composition rounds its prescale, so
// the bound grows with |x| where the prescale is a multiply.
static bool composite_accurate(int op, float x, float y) {
  double xd = x, r, tol;
  switch (op) {
  case OP_EXPF:    r = std::exp(xd);  tol = (2e-7 + 1e-7 * std::fabs(xd)) * std::fabs(r); break;
  case OP_LOGF:    r = std::log(xd);  tol = 1e-6 + 4e-7 * std::fabs(r); break;
  case OP_SINF:    r = std::sin(xd);  tol = 1e-6 + 1e-7 * std::fabs(xd); break;
  case OP_COSF:    r = std::cos(xd);  tol = 1e-6 + 1e-7 * std::fabs(xd); break;
  case OP_SQRTF:   r = std::sqrt(xd); tol = 4e-7 * r; break;
  // tanh is accurate to 2^-11 relative; 1/2 + tanh/2 turns that into 2^-12
  // absolute (small sigmoid values are not relatively accurate).
  case OP_SIGMOID: r = 1.0 / (1.0 + std::exp(-xd)); tol = 0x1p-12; break;
  case OP_SILU:    r = xd / (1.0 + std::exp(-xd)); tol = 0x1p-12 * std::max(1.0, std::fabs(xd)); break;
  case OP_GELU:    r = 0.5 * xd * (1.0 + std::tanh(0.7978845608028654 * (xd + 0.044715 * xd * xd * xd)));
                   tol = 0x1p-12 * std::max(1.0, std::fabs(xd)); break;
  default:         return true;
  }
  return std::fabs((double)y - r) <= tol;
}

static int run_op(int op, std::mt19937& rng, uint64_t src_addr, uint64_t dst_addr) {
  std::vector<float> h_src(num_points);
  std::vector<float> h_dst(num_points);
  size_t ns = (op <= OP_TANH) ? std::min<size_t>(num_points, sizeof(kSpecials) / sizeof(kSpecials[0])) : 0;
  for (uint32_t i = 0; i < num_points; ++i)
    h_src[i] = (i < ns) ? as_float(kSpecials[i]) : gen_input(op, rng);

  kernel_arg_t arg = {};
  arg.op = op;
  arg.num_points = num_points;
  arg.repeat = repeat;
  arg.src_addr = src_addr;
  arg.dst_addr = dst_addr;

  uint32_t buf_size = num_points * sizeof(float);
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), buf_size, 0, nullptr, nullptr));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t grid_dim[1], block_dim[1];
    RT_CHECK(vx_device_max_occupancy_grid(device, 1, &num_points, grid_dim, block_dim));
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &arg;
    li.args_size    = sizeof(arg);
    li.ndim         = 1;
    li.grid_dim[0]  = grid_dim[0];
    li.block_dim[0] = block_dim[0];
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, buf_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    float x = h_src[i], y = h_dst[i];
    bool ok;
    if (op <= OP_TANH) {
      uint32_t ref = vortex::mufu::eval(op, as_bits(x));
      for (uint32_t r = 1; r < repeat; ++r)
        ref = vortex::mufu::eval(op, ref);
      ok = (as_bits(y) == ref);
      if (ok && repeat == 1)
        ok = raw_accurate(op, x, y);
      if (!ok && errors < 20)
        printf("*** error: %s[%u] x=%a (0x%08x): actual=0x%08x expected=0x%08x\n",
               kOpNames[op], i, x, as_bits(x), as_bits(y), ref);
    } else {
      ok = (repeat > 1) || composite_accurate(op, x, y);
      if (!ok && errors < 20)
        printf("*** error: %s[%u] x=%a: actual=%a\n", kOpNames[op], i, x, y);
    }
    errors += !ok;
  }
  std::cout << kOpNames[op] << ": " << num_points << " points, "
            << errors << " errors" << std::endl;
  return errors;
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_MUFU) == 0) {
    std::cout << "MUFU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint32_t buf_size = num_points * sizeof(float);
  uint64_t src_addr, dst_addr;
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &src_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &dst_addr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::mt19937 rng(0x6d756675);
  int errors = 0;
  for (int op = 0; op < OP_COUNT; ++op) {
    if (selected_op >= 0 && op != selected_op)
      continue;
    errors += run_op(op, rng, src_addr, dst_addr);
  }

  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
