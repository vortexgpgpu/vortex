// Myocyte (Rodinia) — standalone self-checking OpenCL port for Vortex.
//
// Cardiac-myocyte ODE model (91 equations: EC-coupling + CaM/CaMKII/CaN)
// integrated with an adaptive Runge-Kutta-Fehlberg 7(8) solver. The solver runs
// on the host and evaluates the ODE right-hand side once per stage by launching
// the OpenCL kernel (kernel.cl). Correctness is checked against a serial CPU
// golden reference that runs the *same* solver and the *same* RHS math (shared
// verbatim via myocyte_model.h) entirely on the host.
//
// Sizes are kept tiny by default so the test runs quickly under RTL simulation:
// one simulated cell (workload) and a one-millisecond integration interval
// (xmax=1 -> a single adaptive outer step). Both are overridable via getopt.
//
// Device max work-group size = NUM_WARPS*NUM_THREADS = 16. The kernel launches
// global=2*NUMBER_THREADS (=4), local=NUMBER_THREADS (=2): 2 work-groups, well
// under 16.
//
// Type is float throughout (the model runs in single precision on the device).
// No `long` fields cross the host/device boundary. One deliberate ABI fix: the
// kernel `timeinst` argument is passed as float (stock Rodinia declared it int
// while the host passed a float bit pattern — a reinterpret bug).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <CL/opencl.h>

#define fp float
#define EQUATIONS 91
#define PARAMETERS 18
#define NUMBER_THREADS 2   // OpenCL local work-group size (=2, <= 16)

// Host build of the shared model: GLOBAL (the device address-space qualifier)
// expands to nothing here, giving plain-pointer host functions kernel_ecc /
// kernel_cam identical to the ones the device compiles.
//
// Force the shared math calls to their single-precision variants so the host
// golden computes in float, matching the device's float OpenCL math builtins
// (C++ would otherwise promote pow/exp/log/... to double). Scoped here, after
// all system headers; only the model header and the solver code below use them.
#define pow(x, y)   powf((x), (y))
#define exp(x)      expf(x)
#define log(x)      logf(x)
#define log10(x)    log10f(x)
#define sqrt(x)     sqrtf(x)
#define fmod(x, y)  fmodf((x), (y))
#define fabs(x)     fabsf(x)
#define GLOBAL
#include "myocyte_model.h"

#define CL_CHECK(_expr)                                                \
  do {                                                                 \
    cl_int _err = _expr;                                               \
    if (_err == CL_SUCCESS)                                            \
      break;                                                           \
    printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);    \
    cleanup();                                                         \
    exit(-1);                                                          \
  } while (0)

#define CL_CHECK2(_expr)                                               \
  ({                                                                   \
    cl_int _err = CL_INVALID_VALUE;                                    \
    decltype(_expr) _ret = _expr;                                      \
    if (_err != CL_SUCCESS) {                                          \
      printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);  \
      cleanup();                                                       \
      exit(-1);                                                        \
    }                                                                  \
    _ret;                                                              \
  })

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;
  FILE* fh = fopen(filename, "r");
  if (NULL == fh) {
    fprintf(stderr, "Failed to load kernel.\n");
    return -1;
  }
  fseek(fh, 0, SEEK_END);
  long fsize = ftell(fh);
  rewind(fh);
  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fh);
  fclose(fh);
  return 0;
}

// ---- OpenCL handles ---------------------------------------------------------
static cl_device_id device_id = NULL;
static cl_context context = NULL;
static cl_command_queue commandQueue = NULL;
static cl_program program = NULL;
static cl_kernel kernel = NULL;
static cl_mem d_initvalu = NULL;
static cl_mem d_finavalu = NULL;
static cl_mem d_params = NULL;
static cl_mem d_com = NULL;
static uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (d_initvalu) clReleaseMemObject(d_initvalu);
  if (d_finavalu) clReleaseMemObject(d_finavalu);
  if (d_params) clReleaseMemObject(d_params);
  if (d_com) clReleaseMemObject(d_com);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// ---- deterministic initial state and parameters (Rodinia y.txt / params.txt)-
static const fp Y_INIT[EQUATIONS] = {
  1.3705539e-003f,  9.8714218e-001f,  9.9182408e-001f,  6.9968275e-006f,  1.0006761e+000f,
  2.5800244e-002f,  1.5606144e-002f,  4.0110808e-003f,  3.1904893e-001f,  4.0110456e-003f,
  9.9463216e-001f,  8.4705483e-003f,  6.0394891e-003f,  8.8875021e-001f,  8.2057663e-007f,
  1.0271527e-007f,  3.5547650e+000f,  7.7561711e-001f,  9.0782945e-003f,  1.1836625e-001f,
  1.0156121e-002f,  2.5328101e-004f,  2.0087204e-003f,  1.3747360e-001f,  2.2097020e-003f,
  7.7737983e-003f,  1.0124416e-002f,  7.5692786e-002f,  1.1608535e-001f,  1.2509040e+000f,
  5.6109998e-001f,  8.8597621e+000f,  8.8611627e+000f,  8.8615763e+000f,  1.3500000e+002f,
  1.8394727e-004f,  1.0924132e-004f,  8.8935135e-005f,  -8.5719165e+001f,  4.1071583e-001f,
  0.0000000e+000f,  0.0000000e+000f,  1.7483040e+004f,  -1.4677756e+004f,  -1.6869784e+005f,
  1.6590486e+005f,  3.8674011e+002f,  6.0137654e+000f,  2.1665981e-003f,  0.0000000e+000f,
  0.0000000e+000f,  0.0000000e+000f,  5.2910875e-001f,  3.1783490e-002f,  7.1017447e-006f,
  2.8316127e-009f,  2.3444660e-009f,  1.3282361e-004f,  4.3772286e-003f,  1.2450156e-002f,
  3.6005070e+000f,  4.3833441e-002f,  5.4727484e-005f,  8.6054033e-009f,  2.3715335e+000f,
  1.3105350e+001f,  2.7224584e-004f,  1.1056849e-005f,  9.5203944e-006f,  1.3418139e-007f,
  1.4654302e-012f,  8.8079626e-009f,  5.6863655e-004f,  2.1354933e-006f,  5.6653572e-006f,
  1.3743596e-003f,  4.3580316e-002f,  3.8789477e-005f,  5.5612103e-010f,  4.1446000e+000f,
  1.1342721e+000f,  1.5613769e-005f,  7.7666417e-006f,  4.5935002e-008f,  3.0629005e-013f,
  3.2828882e-018f,  2.0356103e-014f,  1.2025051e-004f,  4.4862898e-007f,  1.7078402e-007f,
  8.1327361e-007f,
};

// 16 parameters come from Rodinia params.txt; indices 16 (K) and 17 (Mg) are
// read by kernel_cam but absent from the stock data file (a latent Rodinia bug
// that read uninitialised memory). They are set to physiological constants here
// so the run is fully deterministic; both device and golden use identical
// values, so the self-check is unaffected.
static const fp PARAMS[PARAMETERS] = {
  1.000000e003f,  4.180000e002f,  0.000000e000f,  1.200000e002f,
  3.617508e000f,  9.650000e001f,  5.650000e000f,  2.420000e001f,
  9.951600e-002f, 3.000000e-003f,  5.700000e-001f, 5.650000e000f,
  2.420000e001f,  9.951600e-002f,  3.000000e-003f, 5.700000e-001f,
  1.350000e002f,  // [16] K  (intracellular potassium, mM)
  1.000000e000f,  // [17] Mg (intracellular magnesium, mM)
};

static inline bool is_bad(fp v) {          // NaN or +/-Inf
  return !(v == v) || v > 3.0e38f || v < -3.0e38f;
}

// ---- kernel_fin: host-side finalisation of the RHS (from Rodinia kernel_fin.c)
// Adjusts the ECC Ca states with the CaM Ca-flux (com[]) and applies CaM
// inter-compartment diffusion. Runs on the host in both paths.
static void kernel_fin_cpu(fp* initvalu, int off_ecc, int off_Dyad, int off_SL,
                           int off_Cyt, const fp* parameter, fp* finavalu,
                           fp JCaDyad, fp JCaSL, fp JCaCyt) {
  fp BtotDyad      = parameter[2];
  fp CaMKIItotDyad = parameter[3];

  fp Vmyo  = 2.1454e-11f;
  fp Vdyad = 1.7790e-14f;
  fp VSL   = 6.6013e-13f;
  fp kSLmyo = 8.587e-15f;
  fp k0Boff = 0.0014f;
  fp k0Bon  = k0Boff / 0.2f;
  fp k2Boff = k0Boff / 100.0f;
  fp k2Bon  = k0Bon;
  fp k4Bon  = k0Bon;

  finavalu[off_ecc + 35] = finavalu[off_ecc + 35] + 1e-3f * JCaDyad;
  finavalu[off_ecc + 36] = finavalu[off_ecc + 36] + 1e-3f * JCaSL;
  finavalu[off_ecc + 37] = finavalu[off_ecc + 37] + 1e-3f * JCaCyt;

  fp CaMtotDyad = initvalu[off_Dyad + 0] + initvalu[off_Dyad + 1]
                + initvalu[off_Dyad + 2] + initvalu[off_Dyad + 3]
                + initvalu[off_Dyad + 4] + initvalu[off_Dyad + 5]
                + CaMKIItotDyad * (initvalu[off_Dyad + 6] + initvalu[off_Dyad + 7]
                                 + initvalu[off_Dyad + 8] + initvalu[off_Dyad + 9])
                + initvalu[off_Dyad + 12] + initvalu[off_Dyad + 13]
                + initvalu[off_Dyad + 14];
  fp Bdyad = BtotDyad - CaMtotDyad;
  fp J_cam_dyadSL    = 1e-3f * (k0Boff * initvalu[off_Dyad + 0] - k0Bon * Bdyad * initvalu[off_SL + 0]);
  fp J_ca2cam_dyadSL = 1e-3f * (k2Boff * initvalu[off_Dyad + 1] - k2Bon * Bdyad * initvalu[off_SL + 1]);
  fp J_ca4cam_dyadSL = 1e-3f * (k2Boff * initvalu[off_Dyad + 2] - k4Bon * Bdyad * initvalu[off_SL + 2]);

  fp J_cam_SLmyo    = kSLmyo * (initvalu[off_SL + 0] - initvalu[off_Cyt + 0]);
  fp J_ca2cam_SLmyo = kSLmyo * (initvalu[off_SL + 1] - initvalu[off_Cyt + 1]);
  fp J_ca4cam_SLmyo = kSLmyo * (initvalu[off_SL + 2] - initvalu[off_Cyt + 2]);

  finavalu[off_Dyad + 0] = finavalu[off_Dyad + 0] - J_cam_dyadSL;
  finavalu[off_Dyad + 1] = finavalu[off_Dyad + 1] - J_ca2cam_dyadSL;
  finavalu[off_Dyad + 2] = finavalu[off_Dyad + 2] - J_ca4cam_dyadSL;

  finavalu[off_SL + 0] = finavalu[off_SL + 0] + J_cam_dyadSL * Vdyad / VSL - J_cam_SLmyo / VSL;
  finavalu[off_SL + 1] = finavalu[off_SL + 1] + J_ca2cam_dyadSL * Vdyad / VSL - J_ca2cam_SLmyo / VSL;
  finavalu[off_SL + 2] = finavalu[off_SL + 2] + J_ca4cam_dyadSL * Vdyad / VSL - J_ca4cam_SLmyo / VSL;

  finavalu[off_Cyt + 0] = finavalu[off_Cyt + 0] + J_cam_SLmyo / Vmyo;
  finavalu[off_Cyt + 1] = finavalu[off_Cyt + 1] + J_ca2cam_SLmyo / Vmyo;
  finavalu[off_Cyt + 2] = finavalu[off_Cyt + 2] + J_ca4cam_SLmyo / Vmyo;
}

// ---- RHS evaluators: signature (timeinst, initvalu, params, finavalu) -------
typedef void (*rhs_fn)(fp, fp*, const fp*, fp*);

// Golden reference RHS: EC-coupling + 3x CaM + finalisation, entirely on host.
static void master_cpu(fp timeinst, fp* initvalu, const fp* params, fp* finavalu) {
  fp com[3] = {0, 0, 0};
  fp* p = const_cast<fp*>(params);
  kernel_ecc(timeinst, initvalu, finavalu, 0, p);
  kernel_cam(timeinst, initvalu, finavalu, 46, p, 0, com, 0, initvalu[35] * 1e3f);
  kernel_cam(timeinst, initvalu, finavalu, 61, p, 5, com, 1, initvalu[36] * 1e3f);
  kernel_cam(timeinst, initvalu, finavalu, 76, p, 10, com, 2, initvalu[37] * 1e3f);
  kernel_fin_cpu(initvalu, 0, 46, 61, 76, params, finavalu, com[0], com[1], com[2]);
  for (int i = 0; i < EQUATIONS; i++)
    if (is_bad(finavalu[i])) finavalu[i] = 0.0001f;
}

// Device RHS: EC-coupling + 3x CaM on the Vortex device, finalisation on host.
static void master_dev(fp timeinst, fp* initvalu, const fp* params, fp* finavalu) {
  fp com[3] = {0, 0, 0};
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_initvalu, CL_TRUE, 0,
                                EQUATIONS * sizeof(fp), initvalu, 0, NULL, NULL));
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(float), &timeinst));  // float ABI
  size_t local_work_size = NUMBER_THREADS;
  size_t global_work_size = 2 * NUMBER_THREADS;
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_finavalu, CL_TRUE, 0,
                               EQUATIONS * sizeof(fp), finavalu, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_com, CL_TRUE, 0,
                               3 * sizeof(fp), com, 0, NULL, NULL));
  kernel_fin_cpu(initvalu, 0, 46, 61, 76, params, finavalu, com[0], com[1], com[2]);
  for (int i = 0; i < EQUATIONS; i++)
    if (is_bad(finavalu[i])) finavalu[i] = 0.0001f;
}

// ---- Runge-Kutta-Fehlberg 7(8) single step (from Rodinia embedded_fehlberg) --
static void embedded_fehlberg_7_8(rhs_fn master, fp timeinst, fp h, fp* initvalu,
                                  fp* finavalu, fp* error, const fp* parameter) {
  const fp c_1_11 = 41.0f / 840.0f, c6 = 34.0f / 105.0f;
  const fp c_7_8 = 9.0f / 35.0f, c_9_10 = 9.0f / 280.0f;
  const fp a2 = 2.0f / 27.0f, a3 = 1.0f / 9.0f, a4 = 1.0f / 6.0f, a5 = 5.0f / 12.0f;
  const fp a6 = 1.0f / 2.0f, a7 = 5.0f / 6.0f, a8 = 1.0f / 6.0f, a9 = 2.0f / 3.0f, a10 = 1.0f / 3.0f;
  const fp b31 = 1.0f / 36.0f, b32 = 3.0f / 36.0f;
  const fp b41 = 1.0f / 24.0f, b43 = 3.0f / 24.0f;
  const fp b51 = 20.0f / 48.0f, b53 = -75.0f / 48.0f, b54 = 75.0f / 48.0f;
  const fp b61 = 1.0f / 20.0f, b64 = 5.0f / 20.0f, b65 = 4.0f / 20.0f;
  const fp b71 = -25.0f / 108.0f, b74 = 125.0f / 108.0f, b75 = -260.0f / 108.0f, b76 = 250.0f / 108.0f;
  const fp b81 = 31.0f / 300.0f, b85 = 61.0f / 225.0f, b86 = -2.0f / 9.0f, b87 = 13.0f / 900.0f;
  const fp b91 = 2.0f, b94 = -53.0f / 6.0f, b95 = 704.0f / 45.0f, b96 = -107.0f / 9.0f, b97 = 67.0f / 90.0f, b98 = 3.0f;
  const fp b10_1 = -91.0f / 108.0f, b10_4 = 23.0f / 108.0f, b10_5 = -976.0f / 135.0f, b10_6 = 311.0f / 54.0f;
  const fp b10_7 = -19.0f / 60.0f, b10_8 = 17.0f / 6.0f, b10_9 = -1.0f / 12.0f;
  const fp b11_1 = 2383.0f / 4100.0f, b11_4 = -341.0f / 164.0f, b11_5 = 4496.0f / 1025.0f, b11_6 = -301.0f / 82.0f;
  const fp b11_7 = 2133.0f / 4100.0f, b11_8 = 45.0f / 82.0f, b11_9 = 45.0f / 164.0f, b11_10 = 18.0f / 41.0f;
  const fp b12_1 = 3.0f / 205.0f, b12_6 = -6.0f / 41.0f, b12_7 = -3.0f / 205.0f, b12_8 = -3.0f / 41.0f, b12_9 = 3.0f / 41.0f, b12_10 = 6.0f / 41.0f;
  const fp b13_1 = -1777.0f / 4100.0f, b13_4 = -341.0f / 164.0f, b13_5 = 4496.0f / 1025.0f, b13_6 = -289.0f / 82.0f;
  const fp b13_7 = 2193.0f / 4100.0f, b13_8 = 51.0f / 82.0f, b13_9 = 33.0f / 164.0f, b13_10 = 12.0f / 41.0f;
  const fp err_factor = -41.0f / 840.0f;
  fp h2_7 = a2 * h;

  fp iv[EQUATIONS];
  fp fv[13][EQUATIONS];
  int i;

  // Stage 1
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i];
  master(timeinst, iv, parameter, fv[0]);
  // Stage 2
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i] + h2_7 * fv[0][i];
  master(timeinst + h2_7, iv, parameter, fv[1]);
  // Stage 3
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i] + h * (b31 * fv[0][i] + b32 * fv[1][i]);
  master(timeinst + a3 * h, iv, parameter, fv[2]);
  // Stage 4
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i] + h * (b41 * fv[0][i] + b43 * fv[2][i]);
  master(timeinst + a4 * h, iv, parameter, fv[3]);
  // Stage 5
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i] + h * (b51 * fv[0][i] + b53 * fv[2][i] + b54 * fv[3][i]);
  master(timeinst + a5 * h, iv, parameter, fv[4]);
  // Stage 6
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i] + h * (b61 * fv[0][i] + b64 * fv[3][i] + b65 * fv[4][i]);
  master(timeinst + a6 * h, iv, parameter, fv[5]);
  // Stage 7
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i] + h * (b71 * fv[0][i] + b74 * fv[3][i] + b75 * fv[4][i] + b76 * fv[5][i]);
  master(timeinst + a7 * h, iv, parameter, fv[6]);
  // Stage 8
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i] + h * (b81 * fv[0][i] + b85 * fv[4][i] + b86 * fv[5][i] + b87 * fv[6][i]);
  master(timeinst + a8 * h, iv, parameter, fv[7]);
  // Stage 9
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i] + h * (b91 * fv[0][i] + b94 * fv[3][i] + b95 * fv[4][i] + b96 * fv[5][i] + b97 * fv[6][i] + b98 * fv[7][i]);
  master(timeinst + a9 * h, iv, parameter, fv[8]);
  // Stage 10
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i] + h * (b10_1 * fv[0][i] + b10_4 * fv[3][i] + b10_5 * fv[4][i] + b10_6 * fv[5][i] + b10_7 * fv[6][i] + b10_8 * fv[7][i] + b10_9 * fv[8][i]);
  master(timeinst + a10 * h, iv, parameter, fv[9]);
  // Stage 11
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i] + h * (b11_1 * fv[0][i] + b11_4 * fv[3][i] + b11_5 * fv[4][i] + b11_6 * fv[5][i] + b11_7 * fv[6][i] + b11_8 * fv[7][i] + b11_9 * fv[8][i] + b11_10 * fv[9][i]);
  master(timeinst + h, iv, parameter, fv[10]);
  // Stage 12
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i] + h * (b12_1 * fv[0][i] + b12_6 * fv[5][i] + b12_7 * fv[6][i] + b12_8 * fv[7][i] + b12_9 * fv[8][i] + b12_10 * fv[9][i]);
  master(timeinst, iv, parameter, fv[11]);
  // Stage 13
  for (i = 0; i < EQUATIONS; i++) iv[i] = initvalu[i] + h * (b13_1 * fv[0][i] + b13_4 * fv[3][i] + b13_5 * fv[4][i] + b13_6 * fv[5][i] + b13_7 * fv[6][i] + b13_8 * fv[7][i] + b13_9 * fv[8][i] + b13_10 * fv[9][i] + fv[11][i]);
  master(timeinst + h, iv, parameter, fv[12]);

  // 8th-order solution and error estimate
  for (i = 0; i < EQUATIONS; i++)
    finavalu[i] = initvalu[i] + h * (c_1_11 * (fv[0][i] + fv[10][i]) + c6 * fv[5][i]
                + c_7_8 * (fv[6][i] + fv[7][i]) + c_9_10 * (fv[8][i] + fv[9][i]));
  for (i = 0; i < EQUATIONS; i++)
    error[i] = fabs(err_factor * (fv[0][i] + fv[10][i] - fv[11][i] - fv[12][i]));
}

// ---- adaptive-step solver (from Rodinia solver.c) ---------------------------
#define SV_MAX(x, y) ((x) < (y) ? (y) : (x))
#define SV_MIN(x, y) ((x) < (y) ? (x) : (y))
#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125f
#define MAX_SCALE_FACTOR 4.0f

static int solver(rhs_fn master, fp** y, fp* x, int xmax, const fp* params) {
  fp err_exponent = 1.0f / 7.0f;
  fp h, h_init = 1.0f;
  int xmin = 0;
  fp tolerance = 10.0f / (fp)(xmax - xmin);
  fp err[EQUATIONS], scale[EQUATIONS], yy[EQUATIONS];
  int i, j, k;

  x[0] = 0;
  if (xmax < xmin || h_init <= 0.0f) return -2;
  if (xmax == xmin) return 0;
  h = h_init;
  if (h > (xmax - xmin)) h = (fp)xmax - (fp)xmin;

  for (k = 1; k <= xmax; k++) {
    x[k] = k - 1;
    h = h_init;
    fp scale_fina = 1.0f;

    for (j = 0; j < ATTEMPTS; j++) {
      int error = 0, outside = 0;
      fp scale_min = MAX_SCALE_FACTOR;

      embedded_fehlberg_7_8(master, x[k], h, y[k - 1], y[k], err, params);

      for (i = 0; i < EQUATIONS; i++)
        if (err[i] > 0) error = 1;
      if (error != 1) { scale_fina = MAX_SCALE_FACTOR; break; }

      for (i = 0; i < EQUATIONS; i++) {
        yy[i] = (y[k - 1][i] == 0.0f) ? tolerance : fabs(y[k - 1][i]);
        scale[i] = 0.8f * pow(tolerance * yy[i] / err[i], err_exponent);
        if (scale[i] < scale_min) scale_min = scale[i];
      }
      scale_fina = SV_MIN(SV_MAX(scale_min, MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);

      for (i = 0; i < EQUATIONS; i++)
        if (err[i] > (tolerance * yy[i])) outside = 1;
      if (outside == 0) break;

      h = h * scale_fina;
      if (h >= 0.9f) h = 0.9f;
      if (x[k] + h > (fp)xmax) h = (fp)xmax - x[k];
      else if (x[k] + h + 0.5f * h > (fp)xmax) h = 0.5f * h;
    }

    x[k] = x[k] + h;
    if (j >= ATTEMPTS) return -1;
  }
  return 0;
}

// Run the full solver for one cell and return the final state vector.
static int run_cell(rhs_fn master, int xmax, fp* final_out) {
  std::vector<fp> buf((size_t)(xmax + 1) * EQUATIONS);
  std::vector<fp*> y(xmax + 1);
  std::vector<fp> x(xmax + 1);
  for (int k = 0; k <= xmax; k++) y[k] = &buf[(size_t)k * EQUATIONS];
  for (int i = 0; i < EQUATIONS; i++) y[0][i] = Y_INIT[i];

  int status = solver(master, y.data(), x.data(), xmax, PARAMS);
  for (int i = 0; i < EQUATIONS; i++) final_out[i] = y[xmax][i];
  return status;
}

// ---- host arguments ---------------------------------------------------------
static int xmax = 1;       // simulation end time in ms (adaptive steps)
static int workload = 1;   // number of independent simulated cells

static void show_usage() {
  printf("Usage: [-t xmax(ms)] [-w workload_cells] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "t:w:h")) != -1) {
    switch (c) {
    case 't': xmax = atoi(optarg); break;
    case 'w': workload = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (xmax < 1 || workload < 1) {
    printf("Error: xmax and workload must be >= 1\n");
    exit(-1);
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  printf("Myocyte: cells(workload)=%d  xmax=%dms  wg(local)=%d  global=%d  (max wg=16)\n",
         workload, xmax, NUMBER_THREADS, 2 * NUMBER_THREADS);

  // OpenCL setup.
  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  d_initvalu = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, EQUATIONS * sizeof(fp), NULL, &_err));
  d_finavalu = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, EQUATIONS * sizeof(fp), NULL, &_err));
  d_params   = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, PARAMETERS * sizeof(fp), NULL, &_err));
  d_com      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, 3 * sizeof(fp), NULL, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  kernel = CL_CHECK2(clCreateKernel(program, "kernel_gpu_opencl", &_err));
  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  // Constant kernel arguments (buffers). timeinst (arg 0) is set per launch.
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_params, CL_TRUE, 0, PARAMETERS * sizeof(fp), PARAMS, 0, NULL, NULL));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_initvalu));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_finavalu));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_params));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_com));

  // Golden reference (identical for every cell since the initial state is fixed).
  std::vector<fp> ref(EQUATIONS);
  run_cell(master_cpu, xmax, ref.data());

  auto t0 = std::chrono::high_resolution_clock::now();
  int errors = 0;
  std::vector<fp> dev(EQUATIONS);
  for (int w = 0; w < workload; w++) {
    run_cell(master_dev, xmax, dev.data());
    for (int i = 0; i < EQUATIONS; i++) {
      fp a = ref[i], b = dev[i];
      fp tol = 1e-3f + 1e-2f * fabs(a);   // allclose: atol=1e-3, rtol=1e-2
      if (fabs(a - b) > tol) {
        if (errors < 20)
          printf("*** error: cell %d y[%d] expected=%.7e actual=%.7e\n", w, i, a, b);
        ++errors;
      }
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("Elapsed time: %lg ms\n",
         (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

  cleanup();
  if (errors != 0) {
    printf("FAILED! - %d errors\n", errors);
    return errors;
  }
  printf("PASSED!\n");
  return 0;
}
