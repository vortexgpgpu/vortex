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


#include "layout_config.h"
#include "lbm.h"
#include "lbm_macros.h"
#include "main.h"
#include "ocl.h"


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

	params->inpFiles = (char **)malloc(sizeof(char *) * 2);
  params->inpFiles[0] = (char *)malloc(100);
  params->inpFiles[1] = NULL;
  strncpy(params->inpFiles[0], "120_120_150_ldc.of", 100);

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

  MAIN_finalize(&param, &prm);

  LBM_freeGrid((float **)&TEMP_srcGrid);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  return 0;
}

/*############################################################################*/

void MAIN_parseCommandLine(int nArgs, char *arg[], MAIN_Param *param,
                           struct pb_Parameters *params) {
  struct stat fileStat;

  /*if (nArgs < 2) {
    printf("syntax: lbm <time steps>\n");
    exit(1);
  }*/

  param->nTimeSteps = 4; //atoi(arg[1]);

  if (params->inpFiles[0] != NULL) {
    param->obstacleFilename = params->inpFiles[0];

    /*if (stat(param->obstacleFilename, &fileStat) != 0) {
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
    }*/
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
         param->nTimeSteps, param->resultFilename, "store", "lid-driven cavity",
         (param->obstacleFilename == NULL) ? "<none>"
                                           : param->obstacleFilename);
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

	printf("OK+\n");

  // Setup DEVICE datastructures
  OpenCL_LBM_allocateGrid(prm, &OpenCL_srcGrid);
  OpenCL_LBM_allocateGrid(prm, &OpenCL_dstGrid);

	printf("OK-\n");

  // Initialize DEVICE datastructures
  OpenCL_LBM_initializeGrid(prm, OpenCL_srcGrid, TEMP_srcGrid);
  OpenCL_LBM_initializeGrid(prm, OpenCL_dstGrid, TEMP_dstGrid);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  LBM_showGridStatistics(TEMP_srcGrid);

  LBM_freeGrid((float **)&TEMP_srcGrid);
  LBM_freeGrid((float **)&TEMP_dstGrid);

	printf("OK\n");
}

/*############################################################################*/

void MAIN_finalize(const MAIN_Param *param, const OpenCL_Param *prm) {
  LBM_Grid TEMP_srcGrid;

  // Setup TEMP datastructures
  LBM_allocateGrid((float **)&TEMP_srcGrid);

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  OpenCL_LBM_getDeviceGrid(prm, OpenCL_srcGrid, TEMP_srcGrid);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  LBM_showGridStatistics(TEMP_srcGrid);

  LBM_storeVelocityField(TEMP_srcGrid, param->resultFilename, TRUE);

  LBM_freeGrid((float **)&TEMP_srcGrid);
  OpenCL_LBM_freeGrid(OpenCL_srcGrid);
  OpenCL_LBM_freeGrid(OpenCL_dstGrid);

  clReleaseProgram(prm->clProgram);
  clReleaseKernel(prm->clKernel);
  clReleaseCommandQueue(prm->clCommandQueue);
  clReleaseContext(prm->clContext);
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
	prm->clProgram = clCreateProgramWithBuiltInKernels(
      prm->clContext, 1, &prm->clDevice, "performStreamCollide_kernel", &clStatus);
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
