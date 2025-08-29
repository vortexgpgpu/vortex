#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <CL/cl.h>
#include "CL_helper.h"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

/* required precision in degrees	*/
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

// #define WG_SIZE_X (64)
// #define WG_SIZE_Y (4)
// vortex
#define WG_SIZE_X (4)
#define WG_SIZE_Y (4)
float t_chip      = 0.0005;
float chip_height = 0.016;
float chip_width  = 0.016;
float amb_temp    = 80.0;

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
  fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");

  fprintf(stderr, "\t<iteration> - number of iterations\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
  fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile - output file\n");
  exit(1);
}



int main(int argc, char** argv)
{
  if (argc != 7)
    {
      usage(argc,argv);
    }

  char *pfile, *tfile, *ofile;
  int iterations = atoi(argv[3]);

  pfile            = argv[4];
  tfile            = argv[5];
  ofile            = argv[6];
  int numCols      = atoi(argv[1]);
  int numRows      = atoi(argv[1]);
  int layers       = atoi(argv[2]);

  /* calculating parameters*/

  float dx         = chip_height/numRows;
  float dy         = chip_width/numCols;
  float dz         = t_chip/layers;

  float Cap        = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx         = dy / (2.0 * K_SI * t_chip * dx);
  float Ry         = dx / (2.0 * K_SI * t_chip * dy);
  float Rz         = dz / (K_SI * dx * dy);

  float max_slope  = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt         = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce               = cw                                              = stepDivCap/ Rx;
  cn               = cs                                              = stepDivCap/ Ry;
  ct               = cb                                              = stepDivCap/ Rz;

  cc               = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);


  int          err;           
  int size = numCols * numRows * layers;
  float*        tIn      = (float*) calloc(size,sizeof(float));
  float*        pIn      = (float*) calloc(size,sizeof(float));
  float*        tempCopy = (float*)malloc(size * sizeof(float));
  float*        tempOut  = (float*) calloc(size,sizeof(float));
  int i                 = 0;
  int count = size;
  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  size_t global[2];                   
  size_t local[2];
  memcpy(tempCopy,tIn, size * sizeof(float));

  cl_device_id     device_id;     
  cl_context       context;       
  cl_command_queue commands;      
  cl_program       program;       
  cl_kernel        ko_vadd;       

  cl_mem d_a;                     
  cl_mem d_b;                     
  cl_mem d_c;                     
  const char *KernelSource = load_kernel_source("hotspotKernel.cl"); 
  cl_uint numPlatforms;

  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (err != CL_SUCCESS || numPlatforms <= 0)
    {
      printf("Error: Failed to find a platform!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  cl_platform_id Platform[numPlatforms];
  err = clGetPlatformIDs(numPlatforms, Platform, NULL);
  if (err != CL_SUCCESS || numPlatforms <= 0)
    {
      printf("Error: Failed to get the platform!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  for (i = 0; i < numPlatforms; i++)
    {
      err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
      if (err == CL_SUCCESS)
        {
          break;
        } 
    }

  if (device_id == NULL)
    {
      printf("Error: Failed to create a device group!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  err = output_device_info(device_id);

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context)
    {
      printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  commands = clCreateCommandQueue(context, device_id, 0, &err);
  if (!commands)
    {
      printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
  if (!program)
    {
      printf("Error: Failed to create compute program!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t len;
      char buffer[2048];

      printf("Error: Failed to build program executable!\n%s\n", err_code(err));
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      return EXIT_FAILURE;
    }

  ko_vadd = clCreateKernel(program, "hotspotOpt1", &err);
  if (!ko_vadd || err != CL_SUCCESS)
    {
      printf("Error: Failed to create compute kernel!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  d_a  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, NULL);
  d_b  = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(float) * count, NULL, NULL);
  d_c  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, NULL);

  if (!d_a || !d_b || !d_c) 
    {
      printf("Error: Failed to allocate device memory!\n");
      exit(1);
    }    

  err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * count, tIn, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write tIn to source array!\n%s\n", err_code(err));
      exit(1);
    }

  err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * count, pIn, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write pIn to source array!\n%s\n", err_code(err));
      exit(1);
    }

  err = clEnqueueWriteBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * count, tempOut, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write tempOut to source array!\n%s\n", err_code(err));
      exit(1);
    }
  long long start = get_time();
  int j;
  for(j = 0; j < iterations; j++)
    {
      err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_b);
      err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_a);
      err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
      err |= clSetKernelArg(ko_vadd, 3, sizeof(float), &stepDivCap);
      err |= clSetKernelArg(ko_vadd, 4, sizeof(int), &numCols);
      err |= clSetKernelArg(ko_vadd, 5, sizeof(int), &numRows);
      err |= clSetKernelArg(ko_vadd, 6, sizeof(int), &layers);
      err |= clSetKernelArg(ko_vadd, 7, sizeof(float), &ce);
      err |= clSetKernelArg(ko_vadd, 8, sizeof(float), &cw);
      err |= clSetKernelArg(ko_vadd, 9, sizeof(float), &cn);
      err |= clSetKernelArg(ko_vadd, 10, sizeof(float), &cs);
      err |= clSetKernelArg(ko_vadd, 11, sizeof(float), &ct);
      err |= clSetKernelArg(ko_vadd, 12, sizeof(float), &cb);      
      err |= clSetKernelArg(ko_vadd, 13, sizeof(float), &cc);
      if (err != CL_SUCCESS)
        {
          printf("Error: Failed to set kernel arguments!\n");
          exit(1);
        }

      global[0] = numCols;
      global[1] = numRows;

      local[0] = WG_SIZE_X;
      local[1] = WG_SIZE_Y;

      printf("global: %d, %d\n", (int)global[0], (int)global[1]);
      printf("local: %d, %d\n", (int)local[0], (int)local[1]);
    

      err = clEnqueueNDRangeKernel(commands, ko_vadd, 2, NULL, global, local, 0, NULL, NULL);
      if (err)
        {
          printf("Error: Failed to execute kernel!\n%s\n", err_code(err));
          return EXIT_FAILURE;
        }

      cl_mem temp = d_a;
      d_a         = d_c;
      d_c         = temp;
    }

  clFinish(commands);
  long long stop = get_time();
  err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, sizeof(float) * count, tempOut, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to read output array!\n%s\n", err_code(err));
      exit(1);
    }

  float* answer = (float*)calloc(size, sizeof(float));
  computeTempCPU(pIn, tempCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

  float acc = accuracy(tempOut,answer,numRows*numCols*layers);
  float time = (float)((stop - start)/(1000.0 * 1000.0));
  printf("Time: %.3f (s)\n",time);
  printf("Accuracy: %e\n",acc);

  writeoutput(tempOut,numRows,numCols,layers,ofile);
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(ko_vadd);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  return 0;
}

