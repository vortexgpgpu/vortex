
int test_color_kernel() {
  const char KERNEL_NAME[] = "gl_rgba4";

  cl_program program = NULL;
  cl_kernel kernel = NULL;

  unsigned int width = 600, height = 400;
  cl_mem colorBuffer, fragCoord, discard, fragColor;

  size_t kernel_size = sizeof(GLSC2_kernel_color_pocl);

  uint16_t color_init[width*height];
  uint16_t color_out[width*height];
  uint8_t discard_init[width*height];
  float fragColor_init[width*height][4];

  for (uint32_t i=0; i<width*height; ++i) {
    color_init[i] = 0x0000;
    discard_init[i] = 0x00;
    
    fragColor_init[i][0] = 1.f;
    fragColor_init[i][1] = 1.f;
    fragColor_init[i][2] = 1.f;
    fragColor_init[i][3] = 1.f;
  }

  colorBuffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint16_t)*width*height, &color_init, &_err));
  fragCoord = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float[4])*width*height, NULL, &_err));
  discard = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint8_t)*width*height, &discard_init, &_err));
  fragColor = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float[4])*width*height, &fragColor_init, &_err));

  const uint8_t *kernel_bin = GLSC2_kernel_color_pocl;

  program = CL_CHECK2(clCreateProgramWithBinary(
    context, 1, &device_id, &kernel_size, &kernel_bin, NULL, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  
  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(width), &width));	
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(height), &height));	
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(colorBuffer), &colorBuffer));	
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(fragCoord), &fragCoord));	
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(discard), &discard));	
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(fragColor), &fragColor));	
  
  cl_command_queue commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));
  size_t global_work_size = width*height;
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));

  CL_CHECK(clEnqueueReadBuffer(commandQueue, colorBuffer, CL_TRUE, 0, sizeof(uint16_t)*width*height, color_out, 0, NULL, NULL));

  int errors = 0;
  for (int i = 0; i < width*height; ++i) {
    unsigned short ref = 0xFFFF;
    if (color_out[i] != ref) {
      if (errors < 1) 
        printf("*** error: [%d] expected=%08x, actual=%08x\n", i, ref, color_out[i]);
      ++errors;
    }
  }

  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (colorBuffer) clReleaseMemObject(colorBuffer);
  if (fragCoord) clReleaseMemObject(fragCoord);
  if (discard) clReleaseMemObject(discard);  
  if (fragColor) clReleaseMemObject(fragColor);  
  
  return errors;
}

int test_color_kernel_discard_true() {
  const char KERNEL_NAME[] = "gl_rgba4";

  cl_program program = NULL;
  cl_kernel kernel = NULL;

  unsigned int width = 600, height = 400;
  cl_mem colorBuffer, fragCoord, discard, fragColor;

  size_t kernel_size = sizeof(GLSC2_kernel_color_pocl);

  uint16_t color_init[width*height];
  uint16_t color_out[width*height];
  uint8_t discard_init[width*height];
  float fragColor_init[width*height][4];

  for (uint32_t i=0; i<width*height; ++i) {
    color_init[i] = 0x0000;
    discard_init[i] = 0x01;

    fragColor_init[i][0] = 1.f;
    fragColor_init[i][1] = 1.f;
    fragColor_init[i][2] = 1.f;
    fragColor_init[i][3] = 1.f;
  }

  colorBuffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint16_t)*width*height, &color_init, &_err));
  fragCoord = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float[4])*width*height, NULL, &_err));
  discard = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint8_t)*width*height, &discard_init, &_err));
  fragColor = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float[4])*width*height, &fragColor_init, &_err));

  const uint8_t *kernel_bin = GLSC2_kernel_color_pocl;

  program = CL_CHECK2(clCreateProgramWithBinary(
    context, 1, &device_id, &kernel_size, &kernel_bin, NULL, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(width), &width));	
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(height), &height));	
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(colorBuffer), &colorBuffer));	
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(fragCoord), &fragCoord));	
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(discard), &discard));	
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(fragColor), &fragColor));	

  cl_command_queue commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));
  size_t global_work_size = width*height;
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, colorBuffer, CL_TRUE, 0, sizeof(uint16_t)*width*height, color_out, 0, NULL, NULL));

  int errors = 0;
  for (int i = 0; i < width*height; ++i) {
    unsigned short ref = 0x0000;
    if (color_out[i] != ref) {
      if (errors < 1) 
        printf("*** error: [%d] expected=%08x, actual=%08x\n", i, ref, color_out[i]);
      ++errors;
    }
  }

  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (colorBuffer) clReleaseMemObject(colorBuffer);
  if (fragCoord) clReleaseMemObject(fragCoord);
  if (discard) clReleaseMemObject(discard);  
  if (fragColor) clReleaseMemObject(fragColor);

  return errors;
}
