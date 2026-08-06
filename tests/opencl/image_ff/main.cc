// image_ff — standalone self-checking test for Tier-A (fixed-function TEX)
// OpenCL image sampling on Vortex.
//
// Creates a 2D CL_RGBA/UNORM_INT8 power-of-two image seeded from host data and
// samples every texel through a nearest, unnormalized, clamp-to-edge sampler
// (the FF-representable envelope). Built with VX_CFG_EXT_TEX_ENABLE so the
// device reports TEX and the PoCL driver routes read_imagef through the hardware
// vx_tex4 sampler; the same test also passes on a software-only device (the
// driver falls back). The kernel writes the sampled bytes to a global buffer;
// nearest sampling of a UNORM_INT8 image is bit-exact, so the output must equal
// the source pixel for pixel. Prints PASSED!/FAILED! and returns nonzero on
// failure.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <CL/opencl.h>

#define CL_CHECK(_expr)                                              \
  do {                                                              \
    cl_int _err = _expr;                                           \
    if (_err == CL_SUCCESS)                                        \
      break;                                                       \
    printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);\
    cleanup();                                                     \
    exit(-1);                                                      \
  } while (0)

#define CL_CHECK2(_expr)                                            \
  ({                                                               \
    cl_int _err = CL_INVALID_VALUE;                                \
    decltype(_expr) _ret = _expr;                                  \
    if (_err != CL_SUCCESS) {                                      \
      printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);\
      cleanup();                                                   \
      exit(-1);                                                    \
    }                                                             \
    _ret;                                                          \
  })

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
cl_command_queue queue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_sampler sampler = NULL;
cl_mem src_image = NULL;
cl_mem out_buffer = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (queue) clReleaseCommandQueue(queue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (sampler) clReleaseSampler(sampler);
  if (src_image) clReleaseMemObject(src_image);
  if (out_buffer) clReleaseMemObject(out_buffer);
  if (context) clReleaseContext(context);
  if (kernel_bin) free(kernel_bin);
}

int main(int argc, char** argv) {
  int width = 16, height = 16;
  int c;
  while ((c = getopt(argc, argv, "w:h:")) != -1) {
    switch (c) {
    case 'w': width = atoi(optarg); break;
    case 'h': height = atoi(optarg); break;
    default: break;
    }
  }
  // Power-of-two dims are required for the FF fast path; fall back otherwise.
  printf("image_ff: %dx%d RGBA8 (fixed-function TEX)\n", width, height);

  const size_t npixels = (size_t)width * height;
  std::vector<uint8_t> h_src(npixels * 4);
  std::vector<uint8_t> h_out(npixels * 4, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint8_t* p = &h_src[(y * width + x) * 4];
      p[0] = (uint8_t)((x * 7 + y * 13) & 0xff);
      p[1] = (uint8_t)((x * 3 + y * 5 + 17) & 0xff);
      p[2] = (uint8_t)((x ^ y) * 11 & 0xff);
      p[3] = (uint8_t)((x + y) & 0xff);
    }
  }

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  cl_bool image_support = CL_FALSE;
  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support),
                           &image_support, NULL));
  if (!image_support) {
    printf("Device reports no image support.\nFAILED!\n");
    return 1;
  }

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));
  queue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  cl_image_format fmt;
  fmt.image_channel_order = CL_RGBA;
  fmt.image_channel_data_type = CL_UNORM_INT8;

  cl_image_desc desc;
  memset(&desc, 0, sizeof(desc));
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = width;
  desc.image_height = height;

  src_image = CL_CHECK2(clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      &fmt, &desc, h_src.data(), &_err));
  out_buffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        npixels * 4, NULL, &_err));

  sampler = CL_CHECK2(clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE,
                                      CL_FILTER_NEAREST, &_err));

  uint8_t* source = NULL;
  size_t source_size = 0;
  if (read_kernel_file("kernel.cl", &source, &source_size) != 0) {
    cleanup();
    return -1;
  }
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&source, &source_size, &_err));
  free(source);
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  kernel = CL_CHECK2(clCreateKernel(program, "image_ff", &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_image));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_sampler), &sampler));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_buffer));

  size_t global[2] = { (size_t)width, (size_t)height };
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(queue));

  CL_CHECK(clEnqueueReadBuffer(queue, out_buffer, CL_TRUE, 0, npixels * 4,
                               h_out.data(), 0, NULL, NULL));

  int errors = 0;
  for (size_t i = 0; i < npixels * 4; ++i) {
    if (h_out[i] != h_src[i]) {
      if (errors < 8)
        printf("mismatch at byte %zu: got %u expected %u\n", i,
               (unsigned)h_out[i], (unsigned)h_src[i]);
      ++errors;
    }
  }

  cleanup();
  if (errors != 0) {
    printf("Found %d mismatches.\nFAILED!\n", errors);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
