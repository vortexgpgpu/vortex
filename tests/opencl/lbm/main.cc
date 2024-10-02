/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*############################################################################*/

#include <CL/cl.h>
#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>

#include "layout_config.h"
#include "lbm.h"
#include "lbm_macros.h"
#include "main.h"
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

static float* read_output_file(const char* filename, int size) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
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
    if (fread(floats, sizeof(float), size, file) != (size_t)size) {
        fclose(file);
        free(floats);
        perror("Error reading floats from file");
        return NULL;
    }
    // Close the file
    fclose(file);
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

/*############################################################################*/

static cl_mem OpenCL_srcGrid, OpenCL_dstGrid;

/*############################################################################*/

struct pb_TimerSet timers;
int main(int nArgs, char *arg[]) {
  MAIN_Param param;
  int t;

  OpenCL_Param prm;

  pb_InitializeTimerSet(&timers);
  struct pb_Parameters *params;
  params = pb_ReadParameters(&nArgs, arg);

  static LBM_GridPtr TEMP_srcGrid;
  // Setup TEMP datastructures
  LBM_allocateGrid((float **)&TEMP_srcGrid);
  MAIN_parseCommandLine(nArgs, arg, &param, params);
  MAIN_printInfo(&param);

  OpenCL_initialize(params, &prm);
  MAIN_initialize(&param, &prm);

  for (t = 1; t <= param.nTimeSteps; t++) {
    
    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
    OpenCL_LBM_performStreamCollide(&prm, OpenCL_srcGrid, OpenCL_dstGrid);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    LBM_swapGrids(&OpenCL_srcGrid, &OpenCL_dstGrid);

    if ((t & 63) == 0) {
      printf("timestep: %i\n", t);
#if 0
			CUDA_LBM_getDeviceGrid((float**)&CUDA_srcGrid, (float**)&TEMP_srcGrid);
			LBM_showGridStatistics( *TEMP_srcGrid );
#endif
    }
  }

  int errors = MAIN_finalize(&param, &prm);

  LBM_freeGrid((float **)&TEMP_srcGrid);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  return errors;
}

/*############################################################################*/

void MAIN_parseCommandLine(int nArgs, char *arg[], MAIN_Param *param,
                           struct pb_Parameters *params) {
  struct stat fileStat;

  if (nArgs < 2) {
    printf("syntax: lbm <time steps>\n");
    exit(1);
  }

  param->nTimeSteps = atoi(arg[1]);

  if (params->inpFiles[0] != NULL) {
    param->obstacleFilename = params->inpFiles[0];

    if (stat(param->obstacleFilename, &fileStat) != 0) {
      printf("MAIN_parseCommandLine: cannot stat obstacle file '%s'\n",
             param->obstacleFilename);
      exit(1);
    }
    if (fileStat.st_size != SIZE_X * SIZE_Y * SIZE_Z + (SIZE_Y + 1) * SIZE_Z) {
      printf("MAIN_parseCommandLine:\n"
             "\tsize of file '%s' is %i bytes\n"
             "\texpected size is %i bytes\n",
             param->obstacleFilename, (int)fileStat.st_size,
             SIZE_X * SIZE_Y * SIZE_Z + (SIZE_Y + 1) * SIZE_Z);
      exit(1);
    }
  } else
    param->obstacleFilename = NULL;

  param->resultFilename = params->outFile;
}

/*############################################################################*/

void MAIN_printInfo(const MAIN_Param *param) {
  printf("MAIN_printInfo:\n"
         "\tgrid size      : %i x %i x %i = %.2f * 10^6 Cells\n"
         "\tnTimeSteps     : %i\n"
         "\tresult file    : %s\n"
         "\taction         : %s\n"
         "\tsimulation type: %s\n"
         "\tobstacle file  : %s\n\n",
         SIZE_X, SIZE_Y, SIZE_Z, 1e-6 * SIZE_X * SIZE_Y * SIZE_Z,
         param->nTimeSteps, ((param->resultFilename == NULL) ? "<none>" : param->resultFilename), "store", "lid-driven cavity",
         ((param->obstacleFilename == NULL) ? "<none>" : param->obstacleFilename)
  );
}

/*############################################################################*/

void MAIN_initialize(const MAIN_Param *param, const OpenCL_Param *prm) {
  static LBM_Grid TEMP_srcGrid, TEMP_dstGrid;

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  // Setup TEMP datastructures
  LBM_allocateGrid((float **)&TEMP_srcGrid);
  LBM_allocateGrid((float **)&TEMP_dstGrid);
  LBM_initializeGrid(TEMP_srcGrid);
  LBM_initializeGrid(TEMP_dstGrid);

  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  if (param->obstacleFilename != NULL) {
    LBM_loadObstacleFile(TEMP_srcGrid, param->obstacleFilename);
    LBM_loadObstacleFile(TEMP_dstGrid, param->obstacleFilename);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  LBM_initializeSpecialCellsForLDC(TEMP_srcGrid);
  LBM_initializeSpecialCellsForLDC(TEMP_dstGrid);

	pb_SwitchToTimer(&timers, pb_TimerID_COPY);

  // Setup DEVICE datastructures
  OpenCL_LBM_allocateGrid(prm, &OpenCL_srcGrid);
  OpenCL_LBM_allocateGrid(prm, &OpenCL_dstGrid);

  // Initialize DEVICE datastructures
  OpenCL_LBM_initializeGrid(prm, OpenCL_srcGrid, TEMP_srcGrid);
  OpenCL_LBM_initializeGrid(prm, OpenCL_dstGrid, TEMP_dstGrid);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  LBM_showGridStatistics(TEMP_srcGrid);

  LBM_freeGrid((float **)&TEMP_srcGrid);
  LBM_freeGrid((float **)&TEMP_dstGrid);
}

/*############################################################################*/

int MAIN_finalize(const MAIN_Param *param, const OpenCL_Param *prm) {
  LBM_Grid TEMP_srcGrid;

  // Setup TEMP datastructures
  LBM_allocateGrid((float **)&TEMP_srcGrid);

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  OpenCL_LBM_getDeviceGrid(prm, OpenCL_srcGrid, TEMP_srcGrid);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  LBM_showGridStatistics(TEMP_srcGrid);

  float* result_data;
  int dim = 3 * SIZE_X * SIZE_Y * SIZE_Z;
  if (param->resultFilename) {
    LBM_storeVelocityField(TEMP_srcGrid, param->resultFilename, TRUE);
    result_data = read_output_file(param->resultFilename, dim);
  } else {
    LBM_storeVelocityField(TEMP_srcGrid, "result.dat", TRUE);
    result_data = read_output_file("result.dat", dim);
  }

  // verify output
  char* gold_file = replaceFilenameExtension(param->obstacleFilename, ".gold");
  float* gold_data = read_output_file(gold_file, dim);
  if (!gold_data)
    return -1;
  int errors = compare_floats(result_data, gold_data, dim);
  if (errors > 0) {
    printf("FAILED!\n");
  } else {
    printf("PASSED!\n");
  }
  free(result_data);
  free(gold_data);
  free(gold_file);

  LBM_freeGrid((float **)&TEMP_srcGrid);
  OpenCL_LBM_freeGrid(OpenCL_srcGrid);
  OpenCL_LBM_freeGrid(OpenCL_dstGrid);

  clReleaseProgram(prm->clProgram);
  clReleaseKernel(prm->clKernel);
  clReleaseCommandQueue(prm->clCommandQueue);
  clReleaseContext(prm->clContext);

  return errors;
}

void OpenCL_initialize(struct pb_Parameters *p, OpenCL_Param *prm) {
  cl_int clStatus;
  pb_Context *pb_context;
  pb_context = pb_InitOpenCLContext(p);
  if (pb_context == NULL) {
    fprintf(stderr, "Error: No OpenCL platform/device can be found.");
    return;
  }
  prm->clDevice = (cl_device_id)pb_context->clDeviceId;
  prm->clPlatform = (cl_platform_id)pb_context->clPlatformId;
  prm->clContext = (cl_context)pb_context->clContext;

  prm->clCommandQueue = clCreateCommandQueue(
      prm->clContext, prm->clDevice, CL_QUEUE_PROFILING_ENABLE, &clStatus);
  CHECK_ERROR("clCreateCommandQueue")

  pb_SetOpenCL(&(prm->clContext), &(prm->clCommandQueue));

  //const char *clSource[] = {readFile("src/opencl_base/kernel.cl")};
  //prm->clProgram = clCreateProgramWithSource(prm->clContext, 1, clSource, NULL, &clStatus);
  // read kernel binary from file
  uint8_t *kernel_bin = NULL;
  size_t kernel_size;
  //cl_int binary_status = 0;

  clStatus = read_kernel_file("kernel.cl", &kernel_bin, &kernel_size);
  CHECK_ERROR("read_kernel_file")
	prm->clProgram = clCreateProgramWithSource(
        prm->clContext, 1, (const char**)&kernel_bin, &kernel_size, &clStatus);
  CHECK_ERROR("clCreateProgramWithSource")

  //char clOptions[100];
  //sprintf(clOptions, "-I src/opencl_base");
	//clStatus = clBuildProgram(prm->clProgram, 1, &(prm->clDevice), clOptions, NULL, NULL);
	clStatus = clBuildProgram(prm->clProgram, 1, &prm->clDevice, NULL, NULL, NULL);
  CHECK_ERROR("clBuildProgram")

  prm->clKernel =
      clCreateKernel(prm->clProgram, "performStreamCollide_kernel", &clStatus);
  CHECK_ERROR("clCreateKernel")

  //free((void *)clSource[0]);
}
