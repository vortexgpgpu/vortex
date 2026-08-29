// CFD (Rodinia euler3d) — standalone self-checking OpenCL port for Vortex.
//
// Explicit unstructured Euler3D CFD solver. Each element has NNB=4 faces whose
// neighbours are either another element or a boundary sentinel (-1 wing, -2 far
// field). The GPU runs compute_step_factor / compute_flux / time_step over a few
// Runge-Kutta sub-steps. Rather than read an external mesh file, a small
// synthetic unstructured mesh is generated deterministically in-host. The device
// result is checked against a serial CPU reference running the identical euler3d
// math over the same mesh and initial state.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <CL/opencl.h>

// Mesh / solver constants (must match kernel.cl).
#define GAMMA 1.4f
#define NDIM 3
#define NNB 4
#define RK 3   // 3rd order Runge-Kutta
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
#define NVAR (VAR_DENSITY_ENERGY + 1)

// Comparison tolerance. The device compiler may contract a*b+c into fused
// multiply-adds where the host does not, so the flux math is not bit-exact
// host-vs-device; a relative tolerance absorbs the difference.
#define REL_TOL 1e-3f
#define ABS_TOL 1e-5f

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

// 12-byte packed float triple matching the device-side FLOAT3 (struct of 3
// floats). Do NOT use cl_float3, which the host API pads to 16 bytes.
typedef struct { float x, y, z; } FLOAT3;

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;
  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);
  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);
  fclose(fp);
  return 0;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel k_step_factor = NULL, k_flux = NULL, k_time_step = NULL;
cl_mem d_areas = NULL, d_esse = NULL, d_normals = NULL;
cl_mem d_variables = NULL, d_old_variables = NULL, d_fluxes = NULL, d_step_factors = NULL;
cl_mem d_ff_variable = NULL;
cl_mem d_ff_fc_density_energy = NULL, d_ff_fc_momentum_x = NULL,
       d_ff_fc_momentum_y = NULL, d_ff_fc_momentum_z = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (k_step_factor) clReleaseKernel(k_step_factor);
  if (k_flux) clReleaseKernel(k_flux);
  if (k_time_step) clReleaseKernel(k_time_step);
  if (program) clReleaseProgram(program);
  if (d_areas) clReleaseMemObject(d_areas);
  if (d_esse) clReleaseMemObject(d_esse);
  if (d_normals) clReleaseMemObject(d_normals);
  if (d_variables) clReleaseMemObject(d_variables);
  if (d_old_variables) clReleaseMemObject(d_old_variables);
  if (d_fluxes) clReleaseMemObject(d_fluxes);
  if (d_step_factors) clReleaseMemObject(d_step_factors);
  if (d_ff_variable) clReleaseMemObject(d_ff_variable);
  if (d_ff_fc_density_energy) clReleaseMemObject(d_ff_fc_density_energy);
  if (d_ff_fc_momentum_x) clReleaseMemObject(d_ff_fc_momentum_x);
  if (d_ff_fc_momentum_y) clReleaseMemObject(d_ff_fc_momentum_y);
  if (d_ff_fc_momentum_z) clReleaseMemObject(d_ff_fc_momentum_z);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// Workload parameters (small by default so RTL simulation stays under budget).
// block_size is the OpenCL local work-group size and must not exceed the device
// max (NUM_WARPS*NUM_THREADS = 16 in the default CI config).
static int nel = 32;          // number of real mesh elements
static int iterations = 2;    // outer solver iterations
static int block_size = 16;   // local work-group size (<= 16)

static void show_usage() {
  printf("Usage: [-n elements] [-i iterations] [-b block_size] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:i:b:h")) != -1) {
    switch (c) {
    case 'n': nel = atoi(optarg); break;
    case 'i': iterations = atoi(optarg); break;
    case 'b': block_size = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (nel < 2 || iterations < 1 || block_size < 1 || block_size > 16) {
    printf("Error: invalid parameters (need nel>=2, iterations>=1, 1<=block_size<=16)\n");
    exit(-1);
  }
}

// ---- Serial euler3d math (mirrors kernel.cl exactly) --------------------------

static inline void compute_velocity(float density, FLOAT3 momentum, FLOAT3* velocity) {
  velocity->x = momentum.x / density;
  velocity->y = momentum.y / density;
  velocity->z = momentum.z / density;
}
static inline float compute_speed_sqd(FLOAT3 v) {
  return v.x * v.x + v.y * v.y + v.z * v.z;
}
static inline float compute_pressure(float density, float density_energy, float speed_sqd) {
  return (GAMMA - 1.0f) * (density_energy - 0.5f * density * speed_sqd);
}
static inline float compute_speed_of_sound(float density, float pressure) {
  return sqrtf(GAMMA * pressure / density);
}
static inline void compute_flux_contribution(float density, FLOAT3 momentum, float density_energy,
                                             float pressure, FLOAT3 velocity,
                                             FLOAT3* fc_momentum_x, FLOAT3* fc_momentum_y,
                                             FLOAT3* fc_momentum_z, FLOAT3* fc_density_energy) {
  fc_momentum_x->x = velocity.x * momentum.x + pressure;
  fc_momentum_x->y = velocity.x * momentum.y;
  fc_momentum_x->z = velocity.x * momentum.z;
  fc_momentum_y->x = fc_momentum_x->y;
  fc_momentum_y->y = velocity.y * momentum.y + pressure;
  fc_momentum_y->z = velocity.y * momentum.z;
  fc_momentum_z->x = fc_momentum_x->z;
  fc_momentum_z->y = fc_momentum_y->z;
  fc_momentum_z->z = velocity.z * momentum.z + pressure;
  float de_p = density_energy + pressure;
  fc_density_energy->x = velocity.x * de_p;
  fc_density_energy->y = velocity.y * de_p;
  fc_density_energy->z = velocity.z * de_p;
}

static void cpu_compute_step_factor(int nelr, const float* variables, const float* areas,
                                    float* step_factors) {
  for (int i = 0; i < nelr; i++) {
    float density = variables[i + VAR_DENSITY * nelr];
    FLOAT3 momentum;
    momentum.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
    momentum.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
    momentum.z = variables[i + (VAR_MOMENTUM + 2) * nelr];
    float density_energy = variables[i + VAR_DENSITY_ENERGY * nelr];
    FLOAT3 velocity;      compute_velocity(density, momentum, &velocity);
    float speed_sqd      = compute_speed_sqd(velocity);
    float pressure       = compute_pressure(density, density_energy, speed_sqd);
    float speed_of_sound = compute_speed_of_sound(density, pressure);
    step_factors[i] = 0.5f / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
  }
}

static void cpu_compute_flux(int nelr, const int* esse, const float* normals,
                             const float* variables, const float* ff_variable, float* fluxes,
                             const FLOAT3* ff_fc_density_energy, const FLOAT3* ff_fc_momentum_x,
                             const FLOAT3* ff_fc_momentum_y, const FLOAT3* ff_fc_momentum_z) {
  const float smoothing_coefficient = 0.2f;
  for (int i = 0; i < nelr; i++) {
    int j, nb;
    FLOAT3 normal; float normal_len; float factor;

    float density_i = variables[i + VAR_DENSITY * nelr];
    FLOAT3 momentum_i;
    momentum_i.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
    momentum_i.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
    momentum_i.z = variables[i + (VAR_MOMENTUM + 2) * nelr];
    float density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];

    FLOAT3 velocity_i;       compute_velocity(density_i, momentum_i, &velocity_i);
    float speed_sqd_i      = compute_speed_sqd(velocity_i);
    float speed_i          = sqrtf(speed_sqd_i);
    float pressure_i       = compute_pressure(density_i, density_energy_i, speed_sqd_i);
    float speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
    FLOAT3 fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z, fc_i_density_energy;
    compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i,
                              &fc_i_momentum_x, &fc_i_momentum_y, &fc_i_momentum_z, &fc_i_density_energy);

    float flux_i_density = 0.0f;
    FLOAT3 flux_i_momentum; flux_i_momentum.x = flux_i_momentum.y = flux_i_momentum.z = 0.0f;
    float flux_i_density_energy = 0.0f;

    FLOAT3 velocity_nb; float density_nb, density_energy_nb; FLOAT3 momentum_nb;
    FLOAT3 fc_nb_momentum_x, fc_nb_momentum_y, fc_nb_momentum_z, fc_nb_density_energy;
    float speed_sqd_nb, speed_of_sound_nb, pressure_nb;

    for (j = 0; j < NNB; j++) {
      nb = esse[i + j * nelr];
      normal.x = normals[i + (j + 0 * NNB) * nelr];
      normal.y = normals[i + (j + 1 * NNB) * nelr];
      normal.z = normals[i + (j + 2 * NNB) * nelr];
      normal_len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

      if (nb >= 0) {
        density_nb = variables[nb + VAR_DENSITY * nelr];
        momentum_nb.x = variables[nb + (VAR_MOMENTUM + 0) * nelr];
        momentum_nb.y = variables[nb + (VAR_MOMENTUM + 1) * nelr];
        momentum_nb.z = variables[nb + (VAR_MOMENTUM + 2) * nelr];
        density_energy_nb = variables[nb + VAR_DENSITY_ENERGY * nelr];
        compute_velocity(density_nb, momentum_nb, &velocity_nb);
        speed_sqd_nb      = compute_speed_sqd(velocity_nb);
        pressure_nb       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
        speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);
        compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb,
                                  &fc_nb_momentum_x, &fc_nb_momentum_y, &fc_nb_momentum_z, &fc_nb_density_energy);

        factor = -normal_len * smoothing_coefficient * 0.5f *
                 (speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
        flux_i_density += factor * (density_i - density_nb);
        flux_i_density_energy += factor * (density_energy_i - density_energy_nb);
        flux_i_momentum.x += factor * (momentum_i.x - momentum_nb.x);
        flux_i_momentum.y += factor * (momentum_i.y - momentum_nb.y);
        flux_i_momentum.z += factor * (momentum_i.z - momentum_nb.z);

        factor = 0.5f * normal.x;
        flux_i_density += factor * (momentum_nb.x + momentum_i.x);
        flux_i_density_energy += factor * (fc_nb_density_energy.x + fc_i_density_energy.x);
        flux_i_momentum.x += factor * (fc_nb_momentum_x.x + fc_i_momentum_x.x);
        flux_i_momentum.y += factor * (fc_nb_momentum_y.x + fc_i_momentum_y.x);
        flux_i_momentum.z += factor * (fc_nb_momentum_z.x + fc_i_momentum_z.x);

        factor = 0.5f * normal.y;
        flux_i_density += factor * (momentum_nb.y + momentum_i.y);
        flux_i_density_energy += factor * (fc_nb_density_energy.y + fc_i_density_energy.y);
        flux_i_momentum.x += factor * (fc_nb_momentum_x.y + fc_i_momentum_x.y);
        flux_i_momentum.y += factor * (fc_nb_momentum_y.y + fc_i_momentum_y.y);
        flux_i_momentum.z += factor * (fc_nb_momentum_z.y + fc_i_momentum_z.y);

        factor = 0.5f * normal.z;
        flux_i_density += factor * (momentum_nb.z + momentum_i.z);
        flux_i_density_energy += factor * (fc_nb_density_energy.z + fc_i_density_energy.z);
        flux_i_momentum.x += factor * (fc_nb_momentum_x.z + fc_i_momentum_x.z);
        flux_i_momentum.y += factor * (fc_nb_momentum_y.z + fc_i_momentum_y.z);
        flux_i_momentum.z += factor * (fc_nb_momentum_z.z + fc_i_momentum_z.z);
      } else if (nb == -1) {   // wing boundary
        flux_i_momentum.x += normal.x * pressure_i;
        flux_i_momentum.y += normal.y * pressure_i;
        flux_i_momentum.z += normal.z * pressure_i;
      } else if (nb == -2) {   // far field boundary
        factor = 0.5f * normal.x;
        flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 0] + momentum_i.x);
        flux_i_density_energy += factor * (ff_fc_density_energy[0].x + fc_i_density_energy.x);
        flux_i_momentum.x += factor * (ff_fc_momentum_x[0].x + fc_i_momentum_x.x);
        flux_i_momentum.y += factor * (ff_fc_momentum_y[0].x + fc_i_momentum_y.x);
        flux_i_momentum.z += factor * (ff_fc_momentum_z[0].x + fc_i_momentum_z.x);

        factor = 0.5f * normal.y;
        flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y);
        flux_i_density_energy += factor * (ff_fc_density_energy[0].y + fc_i_density_energy.y);
        flux_i_momentum.x += factor * (ff_fc_momentum_x[0].y + fc_i_momentum_x.y);
        flux_i_momentum.y += factor * (ff_fc_momentum_y[0].y + fc_i_momentum_y.y);
        flux_i_momentum.z += factor * (ff_fc_momentum_z[0].y + fc_i_momentum_z.y);

        factor = 0.5f * normal.z;
        flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z);
        flux_i_density_energy += factor * (ff_fc_density_energy[0].z + fc_i_density_energy.z);
        flux_i_momentum.x += factor * (ff_fc_momentum_x[0].z + fc_i_momentum_x.z);
        flux_i_momentum.y += factor * (ff_fc_momentum_y[0].z + fc_i_momentum_y.z);
        flux_i_momentum.z += factor * (ff_fc_momentum_z[0].z + fc_i_momentum_z.z);
      }
    }

    fluxes[i + VAR_DENSITY * nelr] = flux_i_density;
    fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x;
    fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y;
    fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z;
    fluxes[i + VAR_DENSITY_ENERGY * nelr] = flux_i_density_energy;
  }
}

static void cpu_time_step(int j, int nelr, const float* old_variables, float* variables,
                          const float* step_factors, const float* fluxes) {
  for (int i = 0; i < nelr; i++) {
    float factor = step_factors[i] / (float)(RK + 1 - j);
    for (int v = 0; v < NVAR; v++)
      variables[i + v * nelr] = old_variables[i + v * nelr] + factor * fluxes[i + v * nelr];
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  // Round the element count up to a multiple of the work-group size; padded
  // elements duplicate a well-formed boundary cell (never referenced as a
  // neighbour), so the solver stays finite.
  int nelr = block_size * ((nel + block_size - 1) / block_size);
  printf("CFD (euler3d): nel=%d nelr=%d NNB=%d iterations=%d block_size=%d\n",
         nel, nelr, NNB, iterations, block_size);

  // Far-field flow conditions (identical to Rodinia euler3d).
  float h_ff_variable[NVAR];
  FLOAT3 h_ff_fc_momentum_x, h_ff_fc_momentum_y, h_ff_fc_momentum_z, h_ff_fc_density_energy;
  {
    const float angle_of_attack = (float)(3.1415926535897931 / 180.0) * deg_angle_of_attack;
    h_ff_variable[VAR_DENSITY] = 1.4f;
    float ff_pressure = 1.0f;
    float ff_speed_of_sound = sqrtf(GAMMA * ff_pressure / h_ff_variable[VAR_DENSITY]);
    float ff_speed = ff_mach * ff_speed_of_sound;
    FLOAT3 ff_velocity;
    ff_velocity.x = ff_speed * cosf(angle_of_attack);
    ff_velocity.y = ff_speed * sinf(angle_of_attack);
    ff_velocity.z = 0.0f;
    h_ff_variable[VAR_MOMENTUM + 0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
    h_ff_variable[VAR_MOMENTUM + 1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
    h_ff_variable[VAR_MOMENTUM + 2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;
    h_ff_variable[VAR_DENSITY_ENERGY] =
        h_ff_variable[VAR_DENSITY] * (0.5f * (ff_speed * ff_speed)) + (ff_pressure / (GAMMA - 1.0f));
    FLOAT3 ff_momentum;
    ff_momentum.x = h_ff_variable[VAR_MOMENTUM + 0];
    ff_momentum.y = h_ff_variable[VAR_MOMENTUM + 1];
    ff_momentum.z = h_ff_variable[VAR_MOMENTUM + 2];
    compute_flux_contribution(h_ff_variable[VAR_DENSITY], ff_momentum,
                              h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity,
                              &h_ff_fc_momentum_x, &h_ff_fc_momentum_y, &h_ff_fc_momentum_z,
                              &h_ff_fc_density_energy);
  }

  // Generate a small synthetic unstructured mesh deterministically (fixed seed).
  // Areas are kept positive and normals small so the explicit solver stays
  // finite over the (short) run. Neighbours mix interior cells with -1 (wing)
  // and -2 (far field) sentinels to exercise all three flux branches.
  std::vector<float> h_areas(nelr);
  std::vector<int> h_esse(nelr * NNB);
  std::vector<float> h_normals(nelr * NDIM * NNB);
  srand(1234);
  auto frand = []() { return (float)rand() / (float)RAND_MAX; };  // [0,1)
  for (int i = 0; i < nel; i++) {
    h_areas[i] = 0.8f + 0.4f * frand();            // [0.8, 1.2]
    for (int j = 0; j < NNB; j++) {
      int r = rand() % 10;
      int nb;
      if (r < 6)       nb = rand() % nel;          // interior neighbour
      else if (r < 8)  nb = -1;                    // wing boundary
      else             nb = -2;                    // far field boundary
      h_esse[i + j * nelr] = nb;
      for (int k = 0; k < NDIM; k++)
        h_normals[i + (j + k * NNB) * nelr] = (frand() * 2.0f - 1.0f) * 0.15f;  // [-0.15,0.15]
    }
  }
  // Pad remaining elements with a benign boundary cell (all-boundary faces,
  // zero normals, unit area) so device reads stay finite; never checked.
  for (int i = nel; i < nelr; i++) {
    h_areas[i] = 1.0f;
    for (int j = 0; j < NNB; j++) {
      h_esse[i + j * nelr] = -1;
      for (int k = 0; k < NDIM; k++)
        h_normals[i + (j + k * NNB) * nelr] = 0.0f;
    }
  }

  // Initial flow variables: every element starts at the far-field state.
  std::vector<float> h_variables(nelr * NVAR);
  for (int i = 0; i < nelr; i++)
    for (int v = 0; v < NVAR; v++)
      h_variables[i + v * nelr] = h_ff_variable[v];

  // ---- OpenCL setup ----------------------------------------------------------
  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));
  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  k_step_factor = CL_CHECK2(clCreateKernel(program, "compute_step_factor", &_err));
  k_flux        = CL_CHECK2(clCreateKernel(program, "compute_flux", &_err));
  k_time_step   = CL_CHECK2(clCreateKernel(program, "time_step", &_err));

  // Device buffers.
  d_areas = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * nelr, NULL, &_err));
  d_esse = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nelr * NNB, NULL, &_err));
  d_normals = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * nelr * NDIM * NNB, NULL, &_err));
  d_variables = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * nelr * NVAR, NULL, &_err));
  d_old_variables = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * nelr * NVAR, NULL, &_err));
  d_fluxes = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * nelr * NVAR, NULL, &_err));
  d_step_factors = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * nelr, NULL, &_err));
  d_ff_variable = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * NVAR, NULL, &_err));
  d_ff_fc_density_energy = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(FLOAT3), NULL, &_err));
  d_ff_fc_momentum_x = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(FLOAT3), NULL, &_err));
  d_ff_fc_momentum_y = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(FLOAT3), NULL, &_err));
  d_ff_fc_momentum_z = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(FLOAT3), NULL, &_err));

  // Upload the mesh, initial flow, and far-field constants.
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_areas, CL_TRUE, 0, sizeof(float) * nelr, h_areas.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_esse, CL_TRUE, 0, sizeof(int) * nelr * NNB, h_esse.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_normals, CL_TRUE, 0, sizeof(float) * nelr * NDIM * NNB, h_normals.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_variables, CL_TRUE, 0, sizeof(float) * nelr * NVAR, h_variables.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_ff_variable, CL_TRUE, 0, sizeof(float) * NVAR, h_ff_variable, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_ff_fc_density_energy, CL_TRUE, 0, sizeof(FLOAT3), &h_ff_fc_density_energy, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_ff_fc_momentum_x, CL_TRUE, 0, sizeof(FLOAT3), &h_ff_fc_momentum_x, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_ff_fc_momentum_y, CL_TRUE, 0, sizeof(FLOAT3), &h_ff_fc_momentum_y, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_ff_fc_momentum_z, CL_TRUE, 0, sizeof(FLOAT3), &h_ff_fc_momentum_z, 0, NULL, NULL));

  size_t local_work_size = block_size;
  size_t global_work_size = ((nelr + block_size - 1) / block_size) * block_size;

  // Static kernel arguments (buffers do not change across iterations).
  CL_CHECK(clSetKernelArg(k_step_factor, 0, sizeof(cl_mem), &d_variables));
  CL_CHECK(clSetKernelArg(k_step_factor, 1, sizeof(cl_mem), &d_areas));
  CL_CHECK(clSetKernelArg(k_step_factor, 2, sizeof(cl_mem), &d_step_factors));
  CL_CHECK(clSetKernelArg(k_step_factor, 3, sizeof(int), &nelr));

  CL_CHECK(clSetKernelArg(k_flux, 0, sizeof(cl_mem), &d_esse));
  CL_CHECK(clSetKernelArg(k_flux, 1, sizeof(cl_mem), &d_normals));
  CL_CHECK(clSetKernelArg(k_flux, 2, sizeof(cl_mem), &d_variables));
  CL_CHECK(clSetKernelArg(k_flux, 3, sizeof(cl_mem), &d_ff_variable));
  CL_CHECK(clSetKernelArg(k_flux, 4, sizeof(cl_mem), &d_fluxes));
  CL_CHECK(clSetKernelArg(k_flux, 5, sizeof(cl_mem), &d_ff_fc_density_energy));
  CL_CHECK(clSetKernelArg(k_flux, 6, sizeof(cl_mem), &d_ff_fc_momentum_x));
  CL_CHECK(clSetKernelArg(k_flux, 7, sizeof(cl_mem), &d_ff_fc_momentum_y));
  CL_CHECK(clSetKernelArg(k_flux, 8, sizeof(cl_mem), &d_ff_fc_momentum_z));
  CL_CHECK(clSetKernelArg(k_flux, 9, sizeof(int), &nelr));

  CL_CHECK(clSetKernelArg(k_time_step, 1, sizeof(int), &nelr));
  CL_CHECK(clSetKernelArg(k_time_step, 2, sizeof(cl_mem), &d_old_variables));
  CL_CHECK(clSetKernelArg(k_time_step, 3, sizeof(cl_mem), &d_variables));
  CL_CHECK(clSetKernelArg(k_time_step, 4, sizeof(cl_mem), &d_step_factors));
  CL_CHECK(clSetKernelArg(k_time_step, 5, sizeof(cl_mem), &d_fluxes));

  // ---- Solve on the device ---------------------------------------------------
  auto time_start = std::chrono::high_resolution_clock::now();
  for (int n = 0; n < iterations; n++) {
    CL_CHECK(clEnqueueCopyBuffer(commandQueue, d_variables, d_old_variables, 0, 0,
                                 sizeof(float) * nelr * NVAR, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, k_step_factor, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
    for (int j = 0; j < RK; j++) {
      CL_CHECK(clEnqueueNDRangeKernel(commandQueue, k_flux, 1, NULL,
                                      &global_work_size, &local_work_size, 0, NULL, NULL));
      CL_CHECK(clSetKernelArg(k_time_step, 0, sizeof(int), &j));
      CL_CHECK(clEnqueueNDRangeKernel(commandQueue, k_time_step, 1, NULL,
                                      &global_work_size, &local_work_size, 0, NULL, NULL));
    }
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<float> h_gpu(nelr * NVAR);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_variables, CL_TRUE, 0,
                               sizeof(float) * nelr * NVAR, h_gpu.data(), 0, NULL, NULL));

  // ---- Serial CPU reference over the identical mesh + initial state ----------
  std::vector<float> ref_variables(h_variables);
  std::vector<float> ref_old(nelr * NVAR), ref_step(nelr), ref_flux(nelr * NVAR);
  for (int n = 0; n < iterations; n++) {
    ref_old = ref_variables;
    cpu_compute_step_factor(nelr, ref_variables.data(), h_areas.data(), ref_step.data());
    for (int j = 0; j < RK; j++) {
      cpu_compute_flux(nelr, h_esse.data(), h_normals.data(), ref_variables.data(),
                       h_ff_variable, ref_flux.data(), &h_ff_fc_density_energy,
                       &h_ff_fc_momentum_x, &h_ff_fc_momentum_y, &h_ff_fc_momentum_z);
      cpu_time_step(j, nelr, ref_old.data(), ref_variables.data(), ref_step.data(), ref_flux.data());
    }
  }

  // Compare the real (non-padded) elements across all flow variables.
  int errors = 0;
  for (int i = 0; i < nel; i++) {
    for (int v = 0; v < NVAR; v++) {
      float expected = ref_variables[i + v * nelr];
      float actual = h_gpu[i + v * nelr];
      float tol = ABS_TOL + REL_TOL * fabsf(expected);
      if (fabsf(actual - expected) > tol) {
        if (errors < 20)
          printf("*** error: elem=%d var=%d expected=%f actual=%f\n", i, v, expected, actual);
        ++errors;
      }
    }
  }

  cleanup();
  if (errors != 0) {
    printf("FAILED! - %d errors\n", errors);
    return errors;
  }
  printf("PASSED!\n");
  return 0;
}
