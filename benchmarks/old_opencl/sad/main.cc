/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <CL/cl.h>
#include <inttypes.h>
#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "OpenCL_common.h"
#include "file.h"
#include "image.h"
#include "sad.h"
#include "sad_kernel.h"

static unsigned short *load_sads(char *filename);
static void write_sads(char *filename, int image_width_macroblocks,
                       int image_height_macroblocks, unsigned short *sads);
static void write_sads_directly(char *filename, int width, int height,
                                unsigned short *sads);

/* FILE I/O */

unsigned short *load_sads(char *filename) {
  FILE *infile;
  unsigned short *sads;
  int w;
  int h;
  int sads_per_block;

  infile = fopen(filename, "r");

  if (!infile) {
    fprintf(stderr, "Cannot find file '%s'\n", filename);
    exit(-1);
  }

  /* Read image dimensions (measured in macroblocks) */
  w = read16u(infile);
  h = read16u(infile);

  /* Read SAD values.  Only interested in the 4x4 SAD values, which are
   * at the end of the file. */
  sads_per_block = MAX_POS_PADDED * (w * h);
  fseek(infile, 25 * sads_per_block * sizeof(unsigned short), SEEK_CUR);

  sads = (unsigned short *)malloc(sads_per_block * 16 * sizeof(unsigned short));
  fread(sads, sizeof(unsigned short), sads_per_block * 16, infile);
  fclose(infile);

  return sads;
}

/* Compare the reference SADs to the expected SADs.
 */
void check_sads(unsigned short *sads_reference, unsigned short *sads_computed,
                int image_size_macroblocks) {
  int block;

  /* Check the 4x4 SAD values.  These are in sads_reference.
   * Ignore the data at the beginning of sads_computed. */
  sads_computed += 25 * MAX_POS_PADDED * image_size_macroblocks;

  for (block = 0; block < image_size_macroblocks; block++) {
    int subblock;

    for (subblock = 0; subblock < 16; subblock++) {
      int sad_index;

      for (sad_index = 0; sad_index < MAX_POS; sad_index++) {
        int index = (block * 16 + subblock) * MAX_POS_PADDED + sad_index;

        if (sads_reference[index] != sads_computed[index]) {
#if 0
		  /* Print exactly where the mismatch was seen */
		  printf("M %3d %2d %4d (%d = %d)\n", block, subblock, sad_index, sads_reference[index], sads_computed[index]);
#else
          goto mismatch;
#endif
        }
      }
    }
  }

  printf("Success.\n");
  return;

mismatch:
  printf("Computed SADs do not match expected values.\n");
}

/* Extract the SAD data for a particular block type for a particular
 * macroblock from the array of SADs of that block type. */
static inline void write_subblocks(FILE *outfile,
                                   unsigned short *subblock_array,
                                   int macroblock, int count) {
  int block;
  int pos;

  for (block = 0; block < count; block++) {
    unsigned short *vec =
        subblock_array + (block + macroblock * count) * MAX_POS_PADDED;

    /* Write all SADs for this sub-block */
    for (pos = 0; pos < MAX_POS; pos++)
      write16u(outfile, *vec++);
  }
}

/* Write some SAD data to a file for output checking.
 *
 * All SAD values for six rows of macroblocks are written.
 * The six rows consist of the top two, middle two, and bottom two image rows.
 */
void write_sads(char *filename, int mb_width, int mb_height,
                unsigned short *sads) {
  FILE *outfile = fopen(filename, "w");
  int mbs = mb_width * mb_height;
  int row_indir;
  int row_indices[6] = {
      0, 1, mb_height / 2 - 1, mb_height / 2, mb_height - 2, mb_height - 1};

  if (outfile == NULL) {
    fprintf(stderr, "Cannot open output file\n");
    exit(-1);
  }

  /* Write the number of output macroblocks */
  write32u(outfile, mb_width * 6);

  /* Write zeros */
  write32u(outfile, 0);

  /* Each row */
  for (row_indir = 0; row_indir < 6; row_indir++) {
    int row = row_indices[row_indir];

    /* Each block in row */
    int block;
    for (block = mb_width * row; block < mb_width * (row + 1); block++) {
      int blocktype;

      /* Write SADs for all sub-block types */
      for (blocktype = 1; blocktype <= 7; blocktype++)
        write_subblocks(outfile, sads + SAD_TYPE_IX(blocktype, mbs), block,
                        SAD_TYPE_CT(blocktype));
    }
  }

  fclose(outfile);
}

/* FILE I/O for debugging */

static void write_sads_directly(char *filename, int width, int height,
                                unsigned short *sads) {
  FILE *f = fopen(filename, "w");
  int n;

  write16u(f, width);
  write16u(f, height);
  for (n = 0; n < 41 * MAX_POS_PADDED * (width * height); n++) {
    write16u(f, sads[n]);
  }
  fclose(f);
}

static void print_test_sad_vector(unsigned short *base, int macroblock,
                                  int count) {
  int n;
  int searchpos = 17 * 33 + 17;
  for (n = 0; n < count; n++)
    printf(" %d", base[(count * macroblock + n) * MAX_POS_PADDED + searchpos]);
}

static void print_test_sads(unsigned short *sads_computed, int mbs) {
  int macroblock = 5;
  int blocktype;

  for (blocktype = 1; blocktype <= 7; blocktype++) {
    printf("%d:", blocktype);
    print_test_sad_vector(sads_computed + SAD_TYPE_IX(blocktype, mbs),
                          macroblock, SAD_TYPE_CT(blocktype));
    puts("\n");
  }
}

/* MAIN */

int main(int argc, char **argv) {
  struct image_i16 *ref_image;
  struct image_i16 *cur_image;
  unsigned short *sads_computed; /* SADs generated by the program */

  int image_size_bytes;
  int image_width_macroblocks, image_height_macroblocks;
  int image_size_macroblocks;

  struct pb_TimerSet timers;
  struct pb_Parameters *params;

  char oclOverhead[] = "OpenCL Overhead";

  pb_InitializeTimerSet(&timers);
  pb_AddSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);

  params = pb_ReadParameters(&argc, argv);
  params->inpFiles = (char **)malloc(sizeof(char *) * 3);
  params->inpFiles[0] = (char *)malloc(100);
  params->inpFiles[1] = (char *)malloc(100);
  params->inpFiles[2] = NULL;
  strncpy(params->inpFiles[0], "reference.bin", 100);
  strncpy(params->inpFiles[1], "frame.bin", 100);

  if (pb_Parameters_CountInputs(params) != 2) {
    fprintf(stderr, "Expecting two input filenames\n");
    exit(-1);
  }

  /* Read input files */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  ref_image = load_image(params->inpFiles[0]);
  cur_image = load_image(params->inpFiles[1]);
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  printf("Ok\n");

  if ((ref_image->width != cur_image->width) ||
      (ref_image->height != cur_image->height)) {
    fprintf(stderr, "Input images must be the same size\n");
    exit(-1);
  }
  if ((ref_image->width % 16) || (ref_image->height % 16)) {
    fprintf(stderr, "Input image size must be an integral multiple of 16\n");
    exit(-1);
  }

  printf("Ok\n");

  /* Compute parameters, allocate memory */
  image_size_bytes = ref_image->width * ref_image->height * sizeof(short);
  image_width_macroblocks = ref_image->width >> 4;
  image_height_macroblocks = ref_image->height >> 4;
  image_size_macroblocks = image_width_macroblocks * image_height_macroblocks;

  sads_computed = (unsigned short *)malloc(
      41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(short));

  // Run the kernel code
  // ************************************************************************

  cl_int ciErrNum;
  cl_command_queue clCommandQueue;

  cl_kernel mb_sad_calc;
  cl_kernel larger_sad_calc_8;
  cl_kernel larger_sad_calc_16;

  cl_mem imgRef;      /* Reference image on the device */
  cl_mem d_cur_image; /* Current image on the device */
  cl_mem d_sads;      /* SADs on the device */

  // x : image_width_macroblocks
  // y : image_height_macroblocks

  pb_Context *pb_context;
  pb_context = pb_InitOpenCLContext(params);
  if (pb_context == NULL) {
    fprintf(stderr, "Error: No OpenCL platform/device can be found.");
    return -1;
  }

  printf("Ok+\n");

  cl_int clStatus;
  cl_device_id clDevice = (cl_device_id)pb_context->clDeviceId;
  cl_platform_id clPlatform = (cl_platform_id)pb_context->clPlatformId;
  cl_context clContext = (cl_context)pb_context->clContext;

  clCommandQueue = clCreateCommandQueue(clContext, clDevice,
                                        CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);

  printf("Ok!\n");

  pb_SetOpenCL(&clContext, &clCommandQueue);

  printf("Ok!\n");

  pb_SwitchToSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);

  // Read Source Code File
  /*size_t program_length;
const char* source_path = "src/opencl_base/kernel.cl";
char* source = oclLoadProgSource(source_path, "", &program_length);
if(!source) {
  fprintf(stderr, "Could not load program source\n"); exit(1);
}

  cl_program clProgram = clCreateProgramWithSource(clContext, 1, (const char
**)&source, &program_length, &ciErrNum);*/
printf("Ok//-\n");
  cl_program clProgram = clCreateProgramWithBuiltInKernels(
      clContext, 1, &clDevice, "mb_sad_calc;larger_sad_calc_8;larger_sad_calc_16", &ciErrNum);
  printf("Ok//+\n");
  OCL_ERRCK_VAR(ciErrNum);

  printf("Ok+\n");

  //free(source);

  // JIT Compilation Options
  char compileOptions[1024];
  //                -cl-nv-verbose
  sprintf(compileOptions, "\
                -D MAX_POS=%u -D CEIL_POS=%u\
                -D POS_PER_THREAD=%u -D MAX_POS_PADDED=%u\
                -D THREADS_W=%u -D THREADS_H=%u\
                -D SEARCH_RANGE=%u -D SEARCH_DIMENSION=%u\
                \0",
          MAX_POS, CEIL(MAX_POS, POS_PER_THREAD), POS_PER_THREAD,
          MAX_POS_PADDED, THREADS_W, THREADS_H, SEARCH_RANGE, SEARCH_DIMENSION);
  printf("options = %s\n", compileOptions);

  OCL_ERRCK_RETVAL(
      clBuildProgram(clProgram, 1, &clDevice, compileOptions, NULL, NULL));

  /*
  char *build_log;
      size_t ret_val_size;
      OCL_ERRCK_RETVAL( clGetProgramBuildInfo(clProgram, clDevice,
  CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size) );
      build_log = (char *)malloc(ret_val_size+1);
      OCL_ERRCK_RETVAL( clGetProgramBuildInfo(clProgram, clDevice,
  CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL) );

      // Null terminate (original writer wasn't sure)
      build_log[ret_val_size] = '\0';

      fprintf(stderr, "%s\n", build_log );
  */

  mb_sad_calc = clCreateKernel(clProgram, "mb_sad_calc", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  larger_sad_calc_8 = clCreateKernel(clProgram, "larger_sad_calc_8", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  larger_sad_calc_16 =
      clCreateKernel(clProgram, "larger_sad_calc_16", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);

  size_t wgSize;
  size_t comp_wgSize[3];
  cl_ulong localMemSize;
  size_t prefwgSizeMult;
  cl_ulong privateMemSize;

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);

  printf("Ok++\n");

#if 0
    cl_image_format img_format;
    img_format.image_channel_order = CL_R;
    img_format.image_channel_data_type = CL_UNSIGNED_INT16;

    /* Transfer reference image to device */
	imgRef = clCreateImage2D(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &img_format, 
                                      ref_image->width /** sizeof(unsigned short)*/, // width
                                      ref_image->height, // height
                                      ref_image->width * sizeof(unsigned short), // row_pitch
                                      ref_image->data, &ciErrNum);
#endif

#if 1
  imgRef = clCreateBuffer(clContext, CL_MEM_READ_ONLY,
                          ref_image->width * ref_image->height *
                              sizeof(unsigned short),
                          NULL, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  OCL_ERRCK_RETVAL(clEnqueueWriteBuffer(clCommandQueue, imgRef, CL_TRUE, 0,
                                        ref_image->width * ref_image->height *
                                            sizeof(unsigned short),
                                        ref_image->data, 0, NULL, NULL));
#else
  imgRef = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          ref_image->width * ref_image->height *
                              sizeof(unsigned short),
                          ref_image->data, &ciErrNum);
  printf("Allocating %d bytes\n",
         ref_image->width * ref_image->height * sizeof(unsigned short));

#endif
  OCL_ERRCK_VAR(ciErrNum);

  /* Allocate SAD data on the device */

  unsigned short *tmpZero = (unsigned short *)calloc(
      41 * MAX_POS_PADDED * image_size_macroblocks, sizeof(unsigned short));

  /*
      size_t max_alloc_size = 0;
      clGetDeviceInfo(clDevice, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                      sizeof(max_alloc_size), &max_alloc_size, NULL);
      if (max_alloc_size < (41 * MAX_POS_PADDED *
              image_size_macroblocks * sizeof(unsigned short))) {
        fprintf(stderr, "Can't allocate sad buffer: max alloc size is %dMB\n",
                (int) (max_alloc_size >> 20));
        exit(-1);
      }
  */

  d_sads = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR,
                          41 * MAX_POS_PADDED * image_size_macroblocks *
                              sizeof(unsigned short),
                          tmpZero, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  free(tmpZero);

  d_cur_image =
      clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     image_size_bytes, cur_image->data, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);

  /* Set Kernel Parameters */

  OCL_ERRCK_RETVAL(
      clSetKernelArg(mb_sad_calc, 0, sizeof(cl_mem), (void *)&d_sads));
  OCL_ERRCK_RETVAL(
      clSetKernelArg(mb_sad_calc, 1, sizeof(cl_mem), (void *)&d_cur_image));
  OCL_ERRCK_RETVAL(
      clSetKernelArg(mb_sad_calc, 2, sizeof(int), &image_width_macroblocks));
  OCL_ERRCK_RETVAL(
      clSetKernelArg(mb_sad_calc, 3, sizeof(int), &image_height_macroblocks));
  OCL_ERRCK_RETVAL(
      clSetKernelArg(mb_sad_calc, 4, sizeof(cl_mem), (void *)&imgRef));

  OCL_ERRCK_RETVAL(
      clSetKernelArg(larger_sad_calc_8, 0, sizeof(cl_mem), (void *)&d_sads));
  OCL_ERRCK_RETVAL(clSetKernelArg(larger_sad_calc_8, 1, sizeof(int),
                                  &image_width_macroblocks));
  OCL_ERRCK_RETVAL(clSetKernelArg(larger_sad_calc_8, 2, sizeof(int),
                                  &image_height_macroblocks));

  OCL_ERRCK_RETVAL(
      clSetKernelArg(larger_sad_calc_16, 0, sizeof(cl_mem), (void *)&d_sads));
  OCL_ERRCK_RETVAL(clSetKernelArg(larger_sad_calc_16, 1, sizeof(int),
                                  &image_width_macroblocks));
  OCL_ERRCK_RETVAL(clSetKernelArg(larger_sad_calc_16, 2, sizeof(int),
                                  &image_height_macroblocks));

  size_t mb_sad_calc_localWorkSize[2] = {
      CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H, 1};
  size_t mb_sad_calc_globalWorkSize[2] = {
      mb_sad_calc_localWorkSize[0] * CEIL(ref_image->width / 4, THREADS_W),
      mb_sad_calc_localWorkSize[1] * CEIL(ref_image->height / 4, THREADS_H)};

  size_t larger_sad_calc_8_localWorkSize[2] = {32, 4};
  size_t larger_sad_calc_8_globalWorkSize[2] = {image_width_macroblocks * 32,
                                                image_height_macroblocks * 4};

  size_t larger_sad_calc_16_localWorkSize[2] = {32, 1};
  size_t larger_sad_calc_16_globalWorkSize[2] = {image_width_macroblocks * 32,
                                                 image_height_macroblocks * 1};

  printf("Ok+++\n");

  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

  /* Run the 4x4 kernel */
  OCL_ERRCK_RETVAL(clEnqueueNDRangeKernel(clCommandQueue, mb_sad_calc, 2, 0,
                                          mb_sad_calc_globalWorkSize,
                                          mb_sad_calc_localWorkSize, 0, 0, 0));

  /* Run the larger-blocks kernels */
  OCL_ERRCK_RETVAL(clEnqueueNDRangeKernel(
      clCommandQueue, larger_sad_calc_8, 2, 0, larger_sad_calc_8_globalWorkSize,
      larger_sad_calc_8_localWorkSize, 0, 0, 0));

  OCL_ERRCK_RETVAL(clEnqueueNDRangeKernel(clCommandQueue, larger_sad_calc_16, 2,
                                          0, larger_sad_calc_16_globalWorkSize,
                                          larger_sad_calc_16_localWorkSize, 0,
                                          0, 0));

  OCL_ERRCK_RETVAL(clFinish(clCommandQueue));
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);

  /* Transfer SAD data to the host */
  OCL_ERRCK_RETVAL(clEnqueueReadBuffer(
      clCommandQueue, d_sads, CL_TRUE, 0,
      41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short),
      sads_computed, 0, NULL, NULL));

  /* Free GPU memory */
  OCL_ERRCK_RETVAL(clReleaseKernel(larger_sad_calc_8));
  OCL_ERRCK_RETVAL(clReleaseKernel(larger_sad_calc_16));
  OCL_ERRCK_RETVAL(clReleaseProgram(clProgram));

  OCL_ERRCK_RETVAL(clReleaseMemObject(d_sads));
  OCL_ERRCK_RETVAL(clReleaseMemObject(imgRef));
  OCL_ERRCK_RETVAL(clReleaseMemObject(d_cur_image));

  OCL_ERRCK_RETVAL(clFinish(clCommandQueue));
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  // ************************************************************************
  // End GPU Code

  /* Print output */
  if (params->outFile) {
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    write_sads(params->outFile, image_width_macroblocks,
               image_height_macroblocks, sads_computed);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  }

#if 0 /* Debugging */
  print_test_sads(sads_computed, image_size_macroblocks);
  write_sads_directly("sad-debug.bin",
		      ref_image->width / 16, ref_image->height / 16,
		      sads_computed);
#endif

  /* Free memory */
  free(sads_computed);
  free_image(ref_image);
  free_image(cur_image);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);

  OCL_ERRCK_RETVAL(clReleaseCommandQueue(clCommandQueue));
  OCL_ERRCK_RETVAL(clReleaseContext(clContext));

  pb_DestroyTimerSet(&timers);

  return 0;
}
