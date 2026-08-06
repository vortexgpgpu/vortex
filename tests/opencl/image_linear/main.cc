// image_linear — validates read_imagef with CLK_FILTER_LINEAR (bilinear) on the
// Vortex software sampler. Samples at horizontal midpoints and checks the result
// against a CPU 50/50 average reference (clamp-to-edge). PASSED!/FAILED! + exit.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <vector>
#include <CL/opencl.h>

#define CL_CHECK(_expr) do { cl_int _err = _expr; if (_err == CL_SUCCESS) break; \
  printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); cleanup(); exit(-1); } while (0)
#define CL_CHECK2(_expr) ({ cl_int _err = CL_INVALID_VALUE; decltype(_expr) _ret = _expr; \
  if (_err != CL_SUCCESS) { printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); cleanup(); exit(-1); } _ret; })

static int read_kernel_file(const char* fn, uint8_t** data, size_t* size) {
  FILE* fp = fopen(fn, "r"); if (!fp) { fprintf(stderr, "Failed to load kernel.\n"); return -1; }
  fseek(fp, 0, SEEK_END); long fs = ftell(fp); rewind(fp);
  *data = (uint8_t*)malloc(fs); *size = fread(*data, 1, fs, fp); fclose(fp); return 0;
}

cl_device_id device_id = NULL; cl_context context = NULL; cl_command_queue queue = NULL;
cl_program program = NULL; cl_kernel kernel = NULL; cl_sampler sampler = NULL;
cl_mem src_image = NULL, out_buf = NULL;
static void cleanup() {
  if (queue) clReleaseCommandQueue(queue); if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program); if (sampler) clReleaseSampler(sampler);
  if (src_image) clReleaseMemObject(src_image); if (out_buf) clReleaseMemObject(out_buf);
  if (context) clReleaseContext(context);
}

int main(int argc, char** argv) {
  int W = 16, H = 16, c;
  while ((c = getopt(argc, argv, "w:h:")) != -1) { if (c=='w') W=atoi(optarg); else if (c=='h') H=atoi(optarg); }
  printf("image_linear: %dx%d RGBA8\n", W, H);

  std::vector<uint8_t> h_src((size_t)W*H*4);
  for (int y=0;y<H;++y) for (int x=0;x<W;++x) { uint8_t* p=&h_src[(y*W+x)*4];
    p[0]=(uint8_t)((x*7+y*13)&0xff); p[1]=(uint8_t)((x*3+17)&0xff); p[2]=(uint8_t)((y*5)&0xff); p[3]=(uint8_t)((x+y)&0xff); }

  cl_platform_id pf; CL_CHECK(clGetPlatformIDs(1,&pf,NULL));
  CL_CHECK(clGetDeviceIDs(pf,CL_DEVICE_TYPE_DEFAULT,1,&device_id,NULL));
  cl_bool img=CL_FALSE; CL_CHECK(clGetDeviceInfo(device_id,CL_DEVICE_IMAGE_SUPPORT,sizeof(img),&img,NULL));
  if (!img) { printf("no image support.\nFAILED!\n"); return 1; }
  context = CL_CHECK2(clCreateContext(NULL,1,&device_id,NULL,NULL,&_err));
  queue = CL_CHECK2(clCreateCommandQueue(context,device_id,0,&_err));

  cl_image_format fmt = { CL_RGBA, CL_UNORM_INT8 };
  cl_image_desc desc; memset(&desc,0,sizeof(desc));
  desc.image_type=CL_MEM_OBJECT_IMAGE2D; desc.image_width=W; desc.image_height=H;
  src_image = CL_CHECK2(clCreateImage(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,&fmt,&desc,h_src.data(),&_err));
  out_buf = CL_CHECK2(clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(float)*W*H*4,NULL,&_err));
  sampler = CL_CHECK2(clCreateSampler(context,CL_FALSE,CL_ADDRESS_CLAMP_TO_EDGE,CL_FILTER_LINEAR,&_err));

  uint8_t* src=NULL; size_t ss=0; if (read_kernel_file("kernel.cl",&src,&ss)!=0){cleanup();return -1;}
  program = CL_CHECK2(clCreateProgramWithSource(context,1,(const char**)&src,&ss,&_err)); free(src);
  CL_CHECK(clBuildProgram(program,1,&device_id,NULL,NULL,NULL));
  kernel = CL_CHECK2(clCreateKernel(program,"image_linear",&_err));
  CL_CHECK(clSetKernelArg(kernel,0,sizeof(cl_mem),&src_image));
  CL_CHECK(clSetKernelArg(kernel,1,sizeof(cl_mem),&out_buf));
  CL_CHECK(clSetKernelArg(kernel,2,sizeof(cl_sampler),&sampler));
  CL_CHECK(clSetKernelArg(kernel,3,sizeof(int),&W));
  size_t global[2]={(size_t)W,(size_t)H};
  CL_CHECK(clEnqueueNDRangeKernel(queue,kernel,2,NULL,global,NULL,0,NULL,NULL));
  CL_CHECK(clFinish(queue));
  std::vector<float> h_out((size_t)W*H*4);
  CL_CHECK(clEnqueueReadBuffer(queue,out_buf,CL_TRUE,0,sizeof(float)*W*H*4,h_out.data(),0,NULL,NULL));

  int errors=0;
  for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
    int x1 = (x+1 < W) ? x+1 : W-1;  // clamp-to-edge
    for (int ch=0; ch<4; ++ch) {
      float a = h_src[(y*W+x)*4+ch]/255.0f, b = h_src[(y*W+x1)*4+ch]/255.0f;
      float expect = 0.5f*a + 0.5f*b;
      float got = h_out[(y*W+x)*4+ch];
      if (fabsf(got-expect) > 2.0f/255.0f) {
        if (errors<8) printf("mismatch (%d,%d).%d got %.4f expect %.4f\n",x,y,ch,got,expect);
        ++errors;
      }
    }
  }
  cleanup();
  if (errors) { printf("Found %d mismatches.\nFAILED!\n",errors); return 1; }
  printf("PASSED!\n"); return 0;
}
