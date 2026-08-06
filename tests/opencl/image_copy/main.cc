// image_copy — standalone self-checking OpenCL image test for Vortex.
//
// Exercises the OpenCL image feature end-to-end on the Vortex PoCL device:
//   - clCreateImage (2D, CL_RGBA / CL_UNORM_INT8) for a read image seeded from
//     host data (CL_MEM_COPY_HOST_PTR) and a write image,
//   - a runtime sampler (clCreateSampler) passed as a kernel argument,
//   - read_imagef + write_imagef inside the kernel (software sampling),
//   - clEnqueueReadImage to read the result back.
// The kernel is an identity copy, so the output must equal the input pixel for
// pixel. Correctness is checked on the host; prints PASSED!/FAILED! and returns
// a nonzero exit code on failure.

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
    }                                                              \
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
cl_mem dst_image = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (queue) clReleaseCommandQueue(queue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (sampler) clReleaseSampler(sampler);
  if (src_image) clReleaseMemObject(src_image);
  if (dst_image) clReleaseMemObject(dst_image);
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
  printf("image_copy: %dx%d RGBA8\n", width, height);

  const size_t npixels = (size_t)width * height;
  std::vector<uint8_t> h_src(npixels * 4);
  std::vector<uint8_t> h_dst(npixels * 4, 0);
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
  dst_image = CL_CHECK2(clCreateImage(context, CL_MEM_WRITE_ONLY, &fmt, &desc, NULL, &_err));

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
  kernel = CL_CHECK2(clCreateKernel(program, "image_copy", &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_image));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_image));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler));

  size_t global[2] = { (size_t)width, (size_t)height };
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(queue));

  size_t origin[3] = { 0, 0, 0 };
  size_t region[3] = { (size_t)width, (size_t)height, 1 };
  CL_CHECK(clEnqueueReadImage(queue, dst_image, CL_TRUE, origin, region, 0, 0,
                              h_dst.data(), 0, NULL, NULL));

  int errors = 0;
  for (size_t i = 0; i < npixels * 4; ++i) {
    if (h_dst[i] != h_src[i]) {
      if (errors < 8)
        printf("mismatch at byte %zu: got %u expected %u\n", i,
               (unsigned)h_dst[i], (unsigned)h_src[i]);
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
