// image_int — validates read_imageui/write_imageui over a CL_UNSIGNED_INT8 RGBA
// image (the unfiltered integer image path). Identity copy, checked bit-exact.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
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
cl_mem src_image = NULL, dst_image = NULL;
static void cleanup() {
  if (queue) clReleaseCommandQueue(queue); if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program); if (sampler) clReleaseSampler(sampler);
  if (src_image) clReleaseMemObject(src_image); if (dst_image) clReleaseMemObject(dst_image);
  if (context) clReleaseContext(context);
}

int main(int argc, char** argv) {
  int W = 16, H = 16, c;
  while ((c = getopt(argc, argv, "w:h:")) != -1) { if (c=='w') W=atoi(optarg); else if (c=='h') H=atoi(optarg); }
  printf("image_int: %dx%d RGBA UINT8\n", W, H);

  std::vector<uint8_t> h_src((size_t)W*H*4), h_dst((size_t)W*H*4, 0);
  for (size_t i=0;i<h_src.size();++i) h_src[i]=(uint8_t)((i*31+7)&0xff);

  cl_platform_id pf; CL_CHECK(clGetPlatformIDs(1,&pf,NULL));
  CL_CHECK(clGetDeviceIDs(pf,CL_DEVICE_TYPE_DEFAULT,1,&device_id,NULL));
  cl_bool img=CL_FALSE; CL_CHECK(clGetDeviceInfo(device_id,CL_DEVICE_IMAGE_SUPPORT,sizeof(img),&img,NULL));
  if (!img) { printf("no image support.\nFAILED!\n"); return 1; }
  context = CL_CHECK2(clCreateContext(NULL,1,&device_id,NULL,NULL,&_err));
  queue = CL_CHECK2(clCreateCommandQueue(context,device_id,0,&_err));

  cl_image_format fmt = { CL_RGBA, CL_UNSIGNED_INT8 };
  cl_image_desc desc; memset(&desc,0,sizeof(desc));
  desc.image_type=CL_MEM_OBJECT_IMAGE2D; desc.image_width=W; desc.image_height=H;
  src_image = CL_CHECK2(clCreateImage(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,&fmt,&desc,h_src.data(),&_err));
  dst_image = CL_CHECK2(clCreateImage(context,CL_MEM_WRITE_ONLY,&fmt,&desc,NULL,&_err));
  sampler = CL_CHECK2(clCreateSampler(context,CL_FALSE,CL_ADDRESS_CLAMP_TO_EDGE,CL_FILTER_NEAREST,&_err));

  uint8_t* src=NULL; size_t ss=0; if (read_kernel_file("kernel.cl",&src,&ss)!=0){cleanup();return -1;}
  program = CL_CHECK2(clCreateProgramWithSource(context,1,(const char**)&src,&ss,&_err)); free(src);
  CL_CHECK(clBuildProgram(program,1,&device_id,NULL,NULL,NULL));
  kernel = CL_CHECK2(clCreateKernel(program,"image_int",&_err));
  CL_CHECK(clSetKernelArg(kernel,0,sizeof(cl_mem),&src_image));
  CL_CHECK(clSetKernelArg(kernel,1,sizeof(cl_mem),&dst_image));
  CL_CHECK(clSetKernelArg(kernel,2,sizeof(cl_sampler),&sampler));
  size_t global[2]={(size_t)W,(size_t)H};
  CL_CHECK(clEnqueueNDRangeKernel(queue,kernel,2,NULL,global,NULL,0,NULL,NULL));
  CL_CHECK(clFinish(queue));
  size_t origin[3]={0,0,0}, region[3]={(size_t)W,(size_t)H,1};
  CL_CHECK(clEnqueueReadImage(queue,dst_image,CL_TRUE,origin,region,0,0,h_dst.data(),0,NULL,NULL));

  int errors=0;
  for (size_t i=0;i<h_src.size();++i) if (h_dst[i]!=h_src[i]) {
    if (errors<8) printf("mismatch byte %zu got %u expect %u\n",i,(unsigned)h_dst[i],(unsigned)h_src[i]); ++errors; }
  cleanup();
  if (errors) { printf("Found %d mismatches.\nFAILED!\n",errors); return 1; }
  printf("PASSED!\n"); return 0;
}
