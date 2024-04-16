int test_viewport_trans() {

  const char KERNEL_NAME[] = "gl_viewport_division";
  uint num_triangle = 3;

  cl_program program = NULL;
  cl_kernel kernel = NULL;

  cl_mem triangleBuffer;

  size_t kernel_size = sizeof(GLSC2_kernel_viewport_division_pocl);
  size_t triangle_size = num_triangle*4;

  float triangle_init[triangle_size]={
    0.0, 1.0, 0.0, 1.0,
    -1.0, -1.0, 0.0, 1.0,
    1.0, -1.0, 0.0, 1.0
  };

  int viewport[4] = {0, 0, 600, 400};

  float depth_range[2] = {0,0};

  float triangle_out[triangle_size];

  triangleBuffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(triangle_init), &triangle_init, &_err));

  const uint8_t *kernel_bin = GLSC2_kernel_viewport_division_pocl;

  program = CL_CHECK2(clCreateProgramWithBinary(
    context, 1, &device_id, &kernel_size, &kernel_bin, NULL, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(float*), &triangleBuffer));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(viewport), &viewport));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(depth_range), &depth_range));

  cl_command_queue commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));
  size_t global_work_size = num_triangle;
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));

  CL_CHECK(clEnqueueReadBuffer(commandQueue, triangleBuffer, CL_TRUE, 0, sizeof(triangle_out), triangle_out, 0, NULL, NULL));

  int errors = 0;
  for (int i = 0; i < triangle_size; i+=4) {
    unsigned short ref = 0xFFFF;
    // if (color_out[i] != ref) {
    //   if (errors < 1) 
    //     printf("*** error: [%d] expected=%08x, actual=%08x\n", i, ref, color_out[i]);
    //   ++errors;
    // }
    printf("vertex %d, x=%f, y=%f, z=%f\n", i>>2, triangle_out[i],triangle_out[i+1],triangle_out[i+2]);
  }

  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (triangleBuffer) clReleaseMemObject(triangleBuffer);
  
  return errors;
}