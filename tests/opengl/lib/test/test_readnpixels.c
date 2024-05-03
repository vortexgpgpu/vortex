
int test_readnpixels() {
  const char KERNEL_NAME[] = "gl_rgba4_rgba8";

  cl_program program = NULL;
  cl_kernel kernel = NULL;

  unsigned int width = 600, height = 400;
  cl_mem buf_in, buf_out;

  size_t kernel_size = sizeof(GLSC2_kernel_readnpixels_pocl);

  uint16_t buf_in_init[width*height];
  uint32_t buf_out_out[width*height];

  for (uint32_t i=0; i<width*height; ++i) {
    buf_in_init[i] = 0xABCD;
  }

  buf_in = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint16_t)*width*height, &buf_in_init, &_err));
  buf_out = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint32_t)*width*height, NULL, &_err));

  const uint8_t *kernel_bin = GLSC2_kernel_readnpixels_pocl;

  program = CL_CHECK2(clCreateProgramWithBinary(
    context, 1, &device_id, &kernel_size, &kernel_bin, NULL, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  
  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  int x = 0;
  int y = 0;

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(buf_in), &buf_in));	
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(buf_out), &buf_out));	
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(x), &x));	
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(y), &y));	
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(width), &width));	
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(height), &height));	
  
  cl_command_queue commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));
  size_t global_work_size = width*height;
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));

  CL_CHECK(clEnqueueReadBuffer(commandQueue, buf_out, CL_TRUE, 0, sizeof(uint32_t)*width*height, buf_out_out, 0, NULL, NULL));

  int errors = 0;
  for (int i = 0; i < width*height; ++i) {
    unsigned int ref = 0xAABBCCDD;
    if (buf_out_out[i] != ref) {
      if (errors < 1) 
        printf("*** error: [%d] expected=%08x, actual=%08x\n", i, ref, buf_out_out[i]);
      ++errors;
    }
  }

  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (buf_in) clReleaseMemObject(buf_in);
  if (buf_out) clReleaseMemObject(buf_out);
  
  return errors;
}
