/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "convert_dataset.h"
#include "file.h"
#include "gpu_info.h"
#include "ocl.h"

static char* replaceFilenameExtension(const char* filename, const char* ext) {
  const char* dot = strrchr(filename, '.');
  int baseLen = dot ? (dot - filename) : strlen(filename);
  char* sz_out = (char*)malloc(baseLen + strlen(ext) + 1);
  if (!sz_out)
    return NULL;
  strncpy(sz_out, filename, baseLen);
  strcpy(sz_out + baseLen, ext);
  return sz_out;
}

static float* read_output_file(const char* filename, int* out_size) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }
    int size;
    // Read the size (number of floats)
    if (fread(&size, sizeof(int), 1, file) != 1) {
        fclose(file);
        perror("Error reading size from file");
        return NULL;
    }
    // Allocate memory for the floats
    float* floats = (float*)malloc(size * sizeof(float));
    if (floats == NULL) {
        fclose(file);
        perror("Memory allocation failed");
        return NULL;
    }
    // Read the float data
    if (fread(floats, sizeof(float), size, file) != size) {
        fclose(file);
        free(floats);
        perror("Error reading floats from file");
        return NULL;
    }
    // Close the file
    fclose(file);
    // If out_size is not NULL, store the size there
    if (out_size != NULL) {
        *out_size = size;
    }
    return floats;
}

static int compare_floats(const float* src, const float* gold, int count) {
  int num_errors = 0;
  float abstol = 0.0f;
  float max_value = 0.0f;
  // Find the maximum magnitude in the gold array for absolute tolerance calculation
  for (int i = 0; i < count; i++) {
    if (fabs(gold[i]) > max_value)
      max_value = fabs(gold[i]);
  }
  // Absolute tolerance is 0.01% of the maximum magnitude of gold array
  abstol = 1e-4 * max_value;
  // Compare each pair of floats
  for (int i = 0; i < count; i++) {
      float diff = fabs(gold[i] - src[i]);
      if (!(diff <= abstol || diff < 0.002 * fabs(gold[i]))) {
          if (num_errors < 10)
              printf("Fail at row %d: (gold) %f != %f (computed)\n", i, gold[i], src[i]);
          num_errors++;
      }
  }
  return num_errors;
}

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return CL_INVALID_VALUE;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
    return CL_INVALID_VALUE;
  }
  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);

  fclose(fp);

  return CL_SUCCESS;
}

static int generate_vector(float *x_vector, int dim) {
  srand(54321);
  int i;
  for (i = 0; i < dim; i++) {
    x_vector[i] = (rand() / (float)RAND_MAX);
  }
  return 0;
}

int main(int argc, char **argv) {
  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;

  printf("CUDA accelerated sparse matrix vector multiplication****\n");
  printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and "
         "Shengzhao Wu<wu14@illinois.edu>\n");
  printf("This version maintained by Chris Rodrigues  ***********\n");
  parameters = pb_ReadParameters(&argc, argv);
  if ((parameters->inpFiles[0] == NULL) || (parameters->inpFiles[1] == NULL)) {
    fprintf(stderr, "Expecting argment -i <matrix-file>,<vector-file>\n");
    exit(-1);
  }

  pb_InitializeTimerSet(&timers);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  // parameters declaration
  cl_int clStatus;
  pb_Context *pb_context;
  pb_context = pb_InitOpenCLContext(parameters);
  if (pb_context == NULL) {
    fprintf(stderr, "Error: No OpenCL platform/device can be found.");
    return -1;
  }

  cl_device_id clDevice = (cl_device_id)pb_context->clDeviceId;
  cl_platform_id clPlatform = (cl_platform_id)pb_context->clPlatformId;
  cl_context clContext = (cl_context)pb_context->clContext;
  cl_command_queue clCommandQueue = clCreateCommandQueue(
      clContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &clStatus);
  CHECK_ERROR("clCreateCommandQueue")

  pb_SetOpenCL(&clContext, &clCommandQueue);

  //const char *clSource[] = {readFile("src/opencl_base/kernel.cl")};
  // cl_program clProgram =
  // clCreateProgramWithSource(clContext,1,clSource,NULL,&clStatus);
   uint8_t *kernel_bin = NULL;
  size_t kernel_size;
  cl_int binary_status = 0;
  cl_program clProgram;

  clStatus = read_kernel_file("kernel.cl", &kernel_bin, &kernel_size);
  CHECK_ERROR("read_kernel_file")
	clProgram = clCreateProgramWithSource(
        clContext, 1, (const char**)&kernel_bin, &kernel_size, &clStatus);
  CHECK_ERROR("clCreateProgramWithSource")

  char clOptions[50];
  sprintf(clOptions, "");
  clStatus = clBuildProgram(clProgram, 1, &clDevice, clOptions, NULL, NULL);
  CHECK_ERROR("clBuildProgram")

  cl_kernel clKernel = clCreateKernel(clProgram, "spmv_jds_naive", &clStatus);
  CHECK_ERROR("clCreateKernel")

  int len;
  int depth;
  int dim;
  int pad = 32;
  int nzcnt_len;

  // host memory allocation
  // matrix
  float *h_data;
  int *h_indices;
  int *h_ptr;
  int *h_perm;
  int *h_nzcnt;
  // vector
  float *h_Ax_vector;
  float *h_x_vector;

  // device memory allocation
  // matrix
  cl_mem d_data;
  cl_mem d_indices;
  cl_mem d_perm;

  // vector
  cl_mem d_Ax_vector;
  cl_mem d_x_vector;

  cl_mem jds_ptr_int;
  cl_mem sh_zcnt_int;

  // load matrix from files
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  // inputData(parameters->inpFiles[0], &len, &depth, &dim,&nzcnt_len,&pad,
  //    &h_data, &h_indices, &h_ptr,
  //    &h_perm, &h_nzcnt);
  int col_count;
  coo_to_jds(parameters->inpFiles[0], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
             1,                       // row padding
             pad,                     // warp size
             1,                       // pack size
             1,                       // is mirrored?
             0,                       // binary matrix
             1,                       // debug level [0:2]
             &h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm, &col_count, &dim,
             &len, &nzcnt_len, &depth);

  //	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  h_Ax_vector = (float *)malloc(sizeof(float) * dim);
  h_x_vector = (float *)malloc(sizeof(float) * dim);

  input_vec(parameters->inpFiles[1], h_x_vector, dim);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  OpenCLDeviceProp clDeviceProp;
  //	clStatus =
  //clGetDeviceInfo(clDevice,CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV,sizeof(cl_uint),&(clDeviceProp.major),NULL);
  // CHECK_ERROR("clGetDeviceInfo")
  //	clStatus =
  //clGetDeviceInfo(clDevice,CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV,sizeof(cl_uint),&(clDeviceProp.minor),NULL);
  //      CHECK_ERROR("clGetDeviceInfo")
  clStatus =
      clGetDeviceInfo(clDevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                      &(clDeviceProp.multiProcessorCount), NULL);
  CHECK_ERROR("clGetDeviceInfo")

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  // memory allocation
  d_data = clCreateBuffer(clContext, CL_MEM_READ_ONLY, len * sizeof(float),
                          NULL, &clStatus);
  CHECK_ERROR("clCreateBuffer")
  d_indices = clCreateBuffer(clContext, CL_MEM_READ_ONLY, len * sizeof(int),
                             NULL, &clStatus);
  CHECK_ERROR("clCreateBuffer")
  d_perm = clCreateBuffer(clContext, CL_MEM_READ_ONLY, dim * sizeof(int), NULL,
                          &clStatus);
  CHECK_ERROR("clCreateBuffer")
  d_x_vector = clCreateBuffer(clContext, CL_MEM_READ_ONLY, dim * sizeof(float),
                              NULL, &clStatus);
  CHECK_ERROR("clCreateBuffer")
  d_Ax_vector = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY,
                               dim * sizeof(float), NULL, &clStatus);
  CHECK_ERROR("clCreateBuffer")
  jds_ptr_int = clCreateBuffer(clContext, CL_MEM_READ_ONLY, 5000 * sizeof(int),
                               NULL, &clStatus);
  CHECK_ERROR("clCreateBuffer")
  sh_zcnt_int = clCreateBuffer(clContext, CL_MEM_READ_ONLY, 5000 * sizeof(int),
                               NULL, &clStatus);
  CHECK_ERROR("clCreateBuffer")

  clMemSet(clCommandQueue, d_Ax_vector, 0, dim * sizeof(float));

  // memory copy
  clStatus = clEnqueueWriteBuffer(clCommandQueue, d_data, CL_FALSE, 0,
                                  len * sizeof(float), h_data, 0, NULL, NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")
  clStatus = clEnqueueWriteBuffer(clCommandQueue, d_indices, CL_FALSE, 0,
                                  len * sizeof(int), h_indices, 0, NULL, NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")
  clStatus = clEnqueueWriteBuffer(clCommandQueue, d_perm, CL_FALSE, 0,
                                  dim * sizeof(int), h_perm, 0, NULL, NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")
  clStatus = clEnqueueWriteBuffer(clCommandQueue, d_x_vector, CL_FALSE, 0,
                                  dim * sizeof(int), h_x_vector, 0, NULL, NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")

  clStatus = clEnqueueWriteBuffer(clCommandQueue, jds_ptr_int, CL_FALSE, 0,
                                  depth * sizeof(int), h_ptr, 0, NULL, NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")
  clStatus =
      clEnqueueWriteBuffer(clCommandQueue, sh_zcnt_int, CL_TRUE, 0,
                           nzcnt_len * sizeof(int), h_nzcnt, 0, NULL, NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  size_t grid = nzcnt_len * pad;
  size_t block = 1;
  /*compute_active_thread(&block, &grid, nzcnt_len, pad, clDeviceProp.major,
                        clDeviceProp.minor, clDeviceProp.multiProcessorCount);*/
  printf("grid size=%ld, block size=%ld, dim=%d\n", grid, block, dim);

  clStatus = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &d_Ax_vector);
  CHECK_ERROR("clSetKernelArg")
  clStatus = clSetKernelArg(clKernel, 1, sizeof(cl_mem), &d_data);
  CHECK_ERROR("clSetKernelArg")
  clStatus = clSetKernelArg(clKernel, 2, sizeof(cl_mem), &d_indices);
  CHECK_ERROR("clSetKernelArg")
  clStatus = clSetKernelArg(clKernel, 3, sizeof(cl_mem), &d_perm);
  CHECK_ERROR("clSetKernelArg")
  clStatus = clSetKernelArg(clKernel, 4, sizeof(cl_mem), &d_x_vector);
  CHECK_ERROR("clSetKernelArg")
  clStatus = clSetKernelArg(clKernel, 5, sizeof(int), &dim);
  CHECK_ERROR("clSetKernelArg")

  clStatus = clSetKernelArg(clKernel, 6, sizeof(cl_mem), &jds_ptr_int);
  CHECK_ERROR("clSetKernelArg")
  clStatus = clSetKernelArg(clKernel, 7, sizeof(cl_mem), &sh_zcnt_int);
  CHECK_ERROR("clSetKernelArg")

  // main execution
  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

  int i;
  for (i = 0; i < 1; i++) {
    clStatus = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 1, NULL, &grid,
                                      &block, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueNDRangeKernel")
  }
  clStatus = clFinish(clCommandQueue);
  CHECK_ERROR("clFinish")

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  // HtoD memory copy
  clStatus = clEnqueueReadBuffer(clCommandQueue, d_Ax_vector, CL_TRUE, 0,
                          dim * sizeof(float), h_Ax_vector, 0, NULL, NULL);
  CHECK_ERROR("clEnqueueReadBuffer")

  clStatus = clReleaseCommandQueue(clCommandQueue);
  clStatus = clReleaseKernel(clKernel);
  clStatus = clReleaseProgram(clProgram);

  clStatus = clReleaseMemObject(d_data);
  clStatus = clReleaseMemObject(d_indices);
  clStatus = clReleaseMemObject(d_perm);
  clStatus = clReleaseMemObject(d_x_vector);
  clStatus = clReleaseMemObject(d_Ax_vector);
  clStatus = clReleaseMemObject(jds_ptr_int);
  clStatus = clReleaseMemObject(sh_zcnt_int);

  CHECK_ERROR("clReleaseMemObject")
  clStatus = clReleaseContext(clContext);
  clStatus = clReleaseDevice(clDevice);

  if (parameters->outFile) {
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    outputData(parameters->outFile, h_Ax_vector, dim);
  }

  // verify output
  int gold_size;
  char* gold_file = replaceFilenameExtension(parameters->inpFiles[0], ".gold");
  float* gold_data = read_output_file(gold_file, &gold_size);
  if (!gold_data)
    return -1;
  if (gold_size != dim) {
    printf("error: gold data size mismatch: current=%d, expected=%d\n", dim, gold_size);
    return -1;
  }
  int errors = compare_floats(h_Ax_vector, gold_data, gold_size);
  if (errors > 0) {
    printf("FAILED!\n");
  } else {
    printf("PASSED!\n");
  }
  free(gold_data);
  free(gold_file);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  //free((void*)clSource[0]);
  if (kernel_bin) free(kernel_bin);

  free(h_data);
  free(h_indices);
  free(h_ptr);
  free(h_perm);
  free(h_nzcnt);
  free(h_Ax_vector);
  free(h_x_vector);
  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return errors;
}
