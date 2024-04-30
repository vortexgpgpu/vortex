int test_rasterization_triangle() {

  const char KERNEL_NAME[] = "gl_rasterization_triangle";
  uint num_triangle = 3;

  unsigned int width = 600, height = 400;
  cl_program program = NULL;
  cl_kernel kernel = NULL;

  cl_mem gl_Positions, primitives, fragCoord, gl_Rasterization, gl_Discard;

  size_t kernel_size = sizeof(GLSC2_kernel_rasterization_triangle_pocl);
  size_t triangle_size = num_triangle*4;

  float gl_Positions_init[triangle_size]={
    300.0, 400.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 1.0,
    600.0, 0.0, 0.0, 1.0
  };

    float attrb_init[triangle_size]={
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0
  };

  int viewport[4] = {0, 0, 600, 400};

  float depth_range[2] = {0,0};

  float fragCoordOut[4*width*height];
  float gl_RasterizationOut[4*width*height];
  char gl_DiscardOut[width*height];

  gl_Positions = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(gl_Positions_init), &gl_Positions_init, &_err));
  primitives = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(attrb_init), &attrb_init, &_err));
  fragCoord = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float[4])*width*height, NULL, &_err));
  gl_Rasterization = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float[4])*width*height, NULL, &_err));
  gl_Discard = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char)*width*height, NULL, &_err));

  const uint8_t *kernel_bin = GLSC2_kernel_rasterization_triangle_pocl;

  program = CL_CHECK2(clCreateProgramWithBinary(
    context, 1, &device_id, &kernel_size, &kernel_bin, NULL, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  int index =0, num_attrs=1;

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(int), &index));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(int), &width));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &height));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &num_attrs));

  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(float*), &gl_Positions));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(float*), &primitives));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(float*), &fragCoord));
  CL_CHECK(clSetKernelArg(kernel, 7, sizeof(float*), &gl_Rasterization));
  CL_CHECK(clSetKernelArg(kernel, 8, sizeof(char*), &gl_Discard));

  cl_command_queue commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));
  size_t global_work_size = width*height;
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));

  CL_CHECK(clEnqueueReadBuffer(commandQueue, fragCoord, CL_TRUE, 0, sizeof(fragCoordOut), fragCoordOut, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, gl_Rasterization, CL_TRUE, 0, sizeof(gl_RasterizationOut), gl_RasterizationOut, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, gl_Discard, CL_TRUE, 0, sizeof(gl_DiscardOut), gl_DiscardOut, 0, NULL, NULL));


  int errors = 0;
  for (int i = 0; i < width*height*4; i+=4) {
    unsigned short ref = 0xFFFF;
    // if (color_out[i] != ref) {
    //   if (errors < 1) 
    //     printf("*** error: [%d] expected=%08x, actual=%08x\n", i, ref, color_out[i]);
    //   ++errors;
    // }
    //if (gl_DiscardOut[i>>2] == 0)
      //printf("fragColor (%d,%d), x=%f, y=%f, z=%f, w = %f\n", ((i>>2)/width), ((i>>2)%width), gl_RasterizationOut[i],gl_RasterizationOut[i+1],gl_RasterizationOut[i+2], gl_RasterizationOut[i+3]);
  }

  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (gl_Positions) clReleaseMemObject(gl_Positions);
  if (primitives) clReleaseMemObject(primitives);
  if (fragCoord) clReleaseMemObject(fragCoord);

  
  return errors;
}
