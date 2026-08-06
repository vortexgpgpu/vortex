// image_ff_linear — standalone self-checking test for Tier-A bilinear (fixed-
// function TEX) OpenCL image sampling on Vortex.
//
// Upsamples an 8x8 CL_RGBA/UNORM_INT8 image to a 16x16 grid through a linear,
// normalized, clamp-to-edge sampler. Built with VX_CFG_EXT_TEX_ENABLE so the
// device reports TEX and the PoCL driver routes read_imagef through the hardware
// vx_tex4 bilinear sampler; the same test also passes on a software-only device.
//
// The FF unit blends in fixed point (8-bit subpixel weights, 8-bit channels)
// while the host reference blends in float, so the two are NOT bit-identical for
// interpolated taps — they agree within a small tolerance. Prints PASSED!/FAILED!
// and returns nonzero on failure.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

// Host reference: bilinear sample of an RGBA8 image at normalized (u,v), with
// clamp-to-edge addressing, matching read_imagef's half-texel convention.
static void ref_bilinear(const uint8_t* img, int w, int h, float u, float v,
                         float out[4]) {
  float fx = u * w - 0.5f;
  float fy = v * h - 0.5f;
  int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
  float ax = fx - x0, ay = fy - y0;
  int xi0 = x0 < 0 ? 0 : (x0 > w - 1 ? w - 1 : x0);
  int xi1 = (x0 + 1) < 0 ? 0 : ((x0 + 1) > w - 1 ? w - 1 : (x0 + 1));
  int yi0 = y0 < 0 ? 0 : (y0 > h - 1 ? h - 1 : y0);
  int yi1 = (y0 + 1) < 0 ? 0 : ((y0 + 1) > h - 1 ? h - 1 : (y0 + 1));
  for (int c = 0; c < 4; ++c) {
    float p00 = img[(yi0 * w + xi0) * 4 + c] * (1.0f / 255.0f);
    float p10 = img[(yi0 * w + xi1) * 4 + c] * (1.0f / 255.0f);
    float p01 = img[(yi1 * w + xi0) * 4 + c] * (1.0f / 255.0f);
    float p11 = img[(yi1 * w + xi1) * 4 + c] * (1.0f / 255.0f);
    float top = p00 * (1 - ax) + p10 * ax;
    float bot = p01 * (1 - ax) + p11 * ax;
    out[c] = top * (1 - ay) + bot * ay;
  }
}

int main(int argc, char** argv) {
  int in_w = 8, in_h = 8;
  int out_w = 16, out_h = 16;
  int c;
  while ((c = getopt(argc, argv, "w:h:")) != -1) {
    switch (c) {
    case 'w': out_w = atoi(optarg); break;
    case 'h': out_h = atoi(optarg); break;
    default: break;
    }
  }
  printf("image_ff_linear: %dx%d -> %dx%d RGBA8 (fixed-function TEX bilinear)\n",
         in_w, in_h, out_w, out_h);

  std::vector<uint8_t> h_src((size_t)in_w * in_h * 4);
  for (int y = 0; y < in_h; ++y) {
    for (int x = 0; x < in_w; ++x) {
      uint8_t* p = &h_src[(y * in_w + x) * 4];
      // Smooth gradients so interpolation is meaningful and easy to eyeball.
      p[0] = (uint8_t)(x * 255 / (in_w - 1));
      p[1] = (uint8_t)(y * 255 / (in_h - 1));
      p[2] = (uint8_t)((x + y) * 255 / (in_w + in_h - 2));
      p[3] = 255;
    }
  }
  const size_t out_px = (size_t)out_w * out_h;
  std::vector<uint8_t> h_out(out_px * 4, 0);

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
  desc.image_width = in_w;
  desc.image_height = in_h;

  src_image = CL_CHECK2(clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      &fmt, &desc, h_src.data(), &_err));
  out_buffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        out_px * 4, NULL, &_err));

  sampler = CL_CHECK2(clCreateSampler(context, CL_TRUE, CL_ADDRESS_CLAMP_TO_EDGE,
                                      CL_FILTER_LINEAR, &_err));

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
  kernel = CL_CHECK2(clCreateKernel(program, "image_ff_linear", &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_image));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_sampler), &sampler));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_buffer));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &out_w));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &out_h));

  size_t global[2] = { (size_t)out_w, (size_t)out_h };
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(queue));

  CL_CHECK(clEnqueueReadBuffer(queue, out_buffer, CL_TRUE, 0, out_px * 4,
                               h_out.data(), 0, NULL, NULL));

  // FF blends in fixed point; the reference in float. Allow a few LSB.
  const int TOL = 3;
  int errors = 0;
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      float ref[4];
      ref_bilinear(h_src.data(), in_w, in_h,
                   (x + 0.5f) / out_w, (y + 0.5f) / out_h, ref);
      for (int cc = 0; cc < 4; ++cc) {
        int g = (int)(ref[cc] * 255.0f + 0.5f);
        int got = h_out[(y * out_w + x) * 4 + cc];
        if (abs(got - g) > TOL) {
          if (errors < 8)
            printf("mismatch at (%d,%d) ch %d: got %d expected %d\n",
                   x, y, cc, got, g);
          ++errors;
        }
      }
    }
  }

  cleanup();
  if (errors != 0) {
    printf("Found %d mismatches (tol %d).\nFAILED!\n", errors, TOL);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
