// codegen — OpenCL kernels that exercise the device compiler and driver:
// by-value struct arguments and float4 register-group moves. Each kernel is
// checked on its own.

#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include "common.h"

#define CL_CHECK(_expr)                                                \
  do {                                                                 \
    cl_int _err = _expr;                                               \
    if (_err == CL_SUCCESS)                                            \
      break;                                                           \
    printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
    exit(-1);                                                          \
  } while (0)

#define CL_CHECK2(_expr)                                               \
  ({                                                                   \
    cl_int _err = CL_INVALID_VALUE;                                    \
    decltype(_expr) _ret = _expr;                                      \
    if (_err != CL_SUCCESS) {                                          \
      printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
      exit(-1);                                                        \
    }                                                                  \
    _ret;                                                              \
  })

static int size = 32;

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
      case 'n': size = atoi(optarg); break;
      default:
        printf("Usage: [-n work-items] [-h]\n");
        exit(c == 'h' ? 0 : -1);
    }
  }
}

static std::vector<char> read_file(const char* path) {
  FILE* fp = fopen(path, "rb");
  if (!fp) { printf("cannot open %s\n", path); exit(-1); }
  fseek(fp, 0, SEEK_END);
  long n = ftell(fp);
  rewind(fp);
  std::vector<char> data(n + 1, 0);
  if (fread(data.data(), 1, n, fp) != (size_t)n) { printf("cannot read %s\n", path); exit(-1); }
  fclose(fp);
  return data;
}

struct Ctx {
  cl_context ctx;
  cl_command_queue q;
  cl_program prog;
};

static size_t pick_local(cl_device_id dev, size_t global) {
  size_t max_wg = 1;
  CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL));
  size_t local = max_wg < 8 ? max_wg : 8;
  while (global % local) --local;
  return local;
}

static int run_byval(Ctx& c, cl_device_id dev, int n) {
  cl_kernel k = CL_CHECK2(clCreateKernel(c.prog, "k_byval", &_err));
  cl_mem out = CL_CHECK2(clCreateBuffer(c.ctx, CL_MEM_WRITE_ONLY, n * BYVAL_OUTS * sizeof(cl_int), NULL, &_err));
  size_t global = n, local = pick_local(dev, global);

  cl_char c0 = -7;
  s12_t s12 = {1000, -12, 5, 77};
  s24_t s24 = {{11, 22, 33, 44, 55}, 9};
  cl_int kk = 0x1234;
  s16_t s16 = {(cl_long_t)0x0000002a00000005LL, 300};
  s1_t s1 = {-3};
  CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &out));
  CL_CHECK(clSetKernelArg(k, 1, sizeof(c0), &c0));
  CL_CHECK(clSetKernelArg(k, 2, sizeof(s12), &s12));
  CL_CHECK(clSetKernelArg(k, 3, local * sizeof(cl_int), NULL));
  CL_CHECK(clSetKernelArg(k, 4, sizeof(s24), &s24));
  CL_CHECK(clSetKernelArg(k, 5, sizeof(kk), &kk));
  CL_CHECK(clSetKernelArg(k, 6, sizeof(s16), &s16));
  CL_CHECK(clSetKernelArg(k, 7, local * sizeof(cl_int), NULL));
  CL_CHECK(clSetKernelArg(k, 8, sizeof(s1), &s1));
  CL_CHECK(clEnqueueNDRangeKernel(c.q, k, 1, NULL, &global, &local, 0, NULL, NULL));
  std::vector<cl_int> h(n * BYVAL_OUTS);
  CL_CHECK(clEnqueueReadBuffer(c.q, out, CL_TRUE, 0, h.size() * sizeof(cl_int), h.data(), 0, NULL, NULL));

  int errors = 0;
  for (int g = 0; g < n; ++g) {
    int lid = g % (int)local;
    int nxt = (lid + 1) % (int)local;
    cl_int ref[BYVAL_OUTS] = {
      c0,
      s12.a + s12.b * 3 + s12.c * 5 + s12.d * 7,
      s24.v[nxt % 5] + s24.tag,
      kk,
      (cl_int)(s16.q >> 32) ^ s16.i,
      (cl_int)s16.q + s16.i,
      s1.c,
      s12.a + g,
    };
    for (int j = 0; j < BYVAL_OUTS; ++j) {
      if (h[g * BYVAL_OUTS + j] != ref[j]) {
        if (errors < 8) printf("*** byval [%d].%d expected=%d actual=%d\n", g, j, ref[j], h[g * BYVAL_OUTS + j]);
        ++errors;
      }
    }
  }
  clReleaseMemObject(out);
  clReleaseKernel(k);
  return errors;
}

struct f4 { float x, y, z, w; };

static f4 sort_elem(f4 r) {
  f4 nr;
  nr.x = (r.x > r.y) ? r.y : r.x;
  nr.y = (r.y > r.x) ? r.y : r.x;
  nr.z = (r.z > r.w) ? r.w : r.z;
  nr.w = (r.w > r.z) ? r.w : r.z;
  r.x = (nr.x > nr.z) ? nr.z : nr.x;
  r.y = (nr.y > nr.w) ? nr.w : nr.y;
  r.z = (nr.z > nr.x) ? nr.z : nr.x;
  r.w = (nr.w > nr.y) ? nr.w : nr.y;
  nr.x = r.x;
  nr.y = (r.y > r.z) ? r.z : r.y;
  nr.z = (r.z > r.y) ? r.z : r.y;
  nr.w = r.w;
  return nr;
}

static f4 get_lowest(f4 a, f4 b) {
  a.x = (a.x < b.w) ? a.x : b.w;
  a.y = (a.y < b.z) ? a.y : b.z;
  a.z = (a.z < b.y) ? a.z : b.y;
  a.w = (a.w < b.x) ? a.w : b.x;
  return a;
}

static f4 get_highest(f4 a, f4 b) {
  b.x = (a.w >= b.x) ? a.w : b.x;
  b.y = (a.z >= b.y) ? a.z : b.y;
  b.z = (a.y >= b.z) ? a.y : b.z;
  b.w = (a.x >= b.w) ? a.x : b.w;
  return b;
}

// Host replay of k_vec4.
static void merge_ref(const f4* input, f4* result, int gid, int run_len) {
  int a_start = gid * 2 * run_len, b_start = a_start + run_len;
  f4* res = result + a_start;
  int aidx = 0, bidx = 0, outidx = 0;
  f4 a = input[a_start], b = input[b_start];
  while (true) {
    f4 next_a = input[a_start + aidx + 1];
    f4 next_b = input[b_start + bidx + 1];
    f4 na = get_lowest(a, b), nb = get_highest(a, b);
    a = sort_elem(na);
    b = sort_elem(nb);
    res[outidx++] = a;
    bool left_a = aidx + 1 < run_len, left_b = bidx + 1 < run_len;
    if (left_a) {
      if (left_b) {
        if (next_a.x < next_b.x) { aidx += 1; a = next_a; }
        else { bidx += 1; a = next_b; }
      } else {
        aidx += 1; a = next_a;
      }
    } else {
      if (left_b) { bidx += 1; a = next_b; }
      else break;
    }
  }
  res[outidx++] = b;
}

static int run_vec4(Ctx& c, cl_device_id dev, int n) {
  cl_kernel k = CL_CHECK2(clCreateKernel(c.prog, "k_vec4", &_err));
  const int run = VEC4_RUN;
  const int elems = n * 2 * run;
  // Each run holds 4*run small integers in ascending order; one float4 of
  // padding covers the merge's look-ahead read past the last run.
  std::vector<f4> in(elems + 1, f4{0, 0, 0, 0});
  unsigned seed = 12345;
  for (int r = 0; r < n * 2; ++r) {
    float v = 0;
    for (int e = 0; e < run; ++e) {
      float* p = &in[r * run + e].x;
      for (int j = 0; j < 4; ++j) {
        seed = seed * 1103515245u + 12345u;
        v += (float)((seed >> 16) % 5);
        p[j] = v;
      }
    }
  }
  cl_mem din = CL_CHECK2(clCreateBuffer(c.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, in.size() * sizeof(f4), in.data(), &_err));
  cl_mem out = CL_CHECK2(clCreateBuffer(c.ctx, CL_MEM_WRITE_ONLY, elems * sizeof(f4), NULL, &_err));
  size_t global = n, local = pick_local(dev, global);
  CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &din));
  CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &out));
  CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_int), &run));
  CL_CHECK(clEnqueueNDRangeKernel(c.q, k, 1, NULL, &global, &local, 0, NULL, NULL));
  std::vector<f4> h(elems);
  CL_CHECK(clEnqueueReadBuffer(c.q, out, CL_TRUE, 0, elems * sizeof(f4), h.data(), 0, NULL, NULL));

  std::vector<f4> ref(elems);
  for (int g = 0; g < n; ++g) merge_ref(in.data(), ref.data(), g, run);
  int errors = 0;
  for (int i = 0; i < elems; ++i) {
    const float* hp = &h[i].x;
    const float* rp = &ref[i].x;
    for (int j = 0; j < 4; ++j) {
      if (hp[j] != rp[j]) {
        if (errors < 8) printf("*** vec4 [%d].%d expected=%g actual=%g\n", i, j, rp[j], hp[j]);
        ++errors;
      }
    }
  }
  clReleaseMemObject(out);
  clReleaseMemObject(din);
  clReleaseKernel(k);
  return errors;
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  printf("codegen: work-items=%d\n", size);

  cl_platform_id platform;
  cl_device_id dev;
  CL_CHECK(clGetPlatformIDs(1, &platform, NULL));
  CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &dev, NULL));
  Ctx c;
  c.ctx = CL_CHECK2(clCreateContext(NULL, 1, &dev, NULL, NULL, &_err));
  c.q = CL_CHECK2(clCreateCommandQueue(c.ctx, dev, 0, &_err));
  std::vector<char> src = read_file("kernel.cl");
  const char* srcp = src.data();
  c.prog = CL_CHECK2(clCreateProgramWithSource(c.ctx, 1, &srcp, NULL, &_err));
  if (clBuildProgram(c.prog, 1, &dev, NULL, NULL, NULL) != CL_SUCCESS) {
    std::vector<char> log(16384);
    clGetProgramBuildInfo(c.prog, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), NULL);
    printf("build failed:\n%s\n", log.data());
    return -1;
  }

  int byval_errors  = run_byval(c, dev, size);
  int vec4_errors   = run_vec4(c, dev, size);

  clReleaseProgram(c.prog);
  clReleaseCommandQueue(c.q);
  clReleaseContext(c.ctx);

  int errors = byval_errors + vec4_errors;
  if (errors) {
    printf("byval: %d errors, vec4: %d errors\nFAILED!\n", byval_errors, vec4_errors);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
