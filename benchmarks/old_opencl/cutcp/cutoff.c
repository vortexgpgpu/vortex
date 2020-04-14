/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <parboil.h>

#include "atom.h"
#include "cutoff.h"
#include "macros.h"
#include "ocl.h"

// OpenCL 1.1 support for int3 is not uniform on all implementations, so
// we use int4 instead.  Only the 'x', 'y', and 'z' fields of xyz are used.
typedef cl_int4 xyz;

//extern "C" int gpu_compute_cutoff_potential_lattice(
int gpu_compute_cutoff_potential_lattice(
    struct pb_TimerSet *timers,
    Lattice *lattice,                  /* the lattice */
    float cutoff,                      /* cutoff distance */
    Atoms *atoms,                      /* array of atoms */
    int verbose,                        /* print info/debug messages */
    struct pb_Parameters *parameters
    )
{
  int nx = lattice->dim.nx;
  int ny = lattice->dim.ny;
  int nz = lattice->dim.nz;
  float xlo = lattice->dim.lo.x;
  float ylo = lattice->dim.lo.y;
  float zlo = lattice->dim.lo.z;
  float h = lattice->dim.h;
  int natoms = atoms->size;
  Atom *atom = atoms->atoms;

  xyz nbrlist[NBRLIST_MAXLEN];
  int nbrlistlen = 0;

  int binHistoFull[BIN_DEPTH+1] = { 0 };   /* clear every array element */
  int binHistoCover[BIN_DEPTH+1] = { 0 };  /* clear every array element */
  int num_excluded = 0;

  int xRegionDim, yRegionDim, zRegionDim;
  int xRegionIndex, yRegionIndex, zRegionIndex;
  int xOffset, yOffset, zOffset;
  int lnx, lny, lnz, lnall;
  float *regionZeroAddr, *thisRegion;
  cl_mem regionZeroCl;
  int index, indexRegion;

  int c;
  xyz binDim;
  int nbins;
  cl_float4 *binBaseAddr, *binZeroAddr;
  cl_mem binBaseCl, binZeroCl;
  int *bincntBaseAddr, *bincntZeroAddr;
  Atoms *extra = NULL;

  cl_mem NbrListLen;
  cl_mem NbrList;

  int i, j, k, n;
  int sum, total;

  float avgFillFull, avgFillCover;
  const float cutoff2 = cutoff * cutoff;
  const float inv_cutoff2 = 1.f / cutoff2;

  size_t gridDim[3], blockDim[3];

  // The "compute" timer should be active upon entry to this function

  /* pad lattice to be factor of 8 in each dimension */
  xRegionDim = (int) ceilf(nx/8.f);
  yRegionDim = (int) ceilf(ny/8.f);
  zRegionDim = (int) ceilf(nz/8.f);

  lnx = 8 * xRegionDim;
  lny = 8 * yRegionDim;
  lnz = 8 * zRegionDim;
  lnall = lnx * lny * lnz;

  /* will receive energies from OpenCL */
  regionZeroAddr = (float *) malloc(lnall * sizeof(float));

  /* create bins */
  c = (int) ceil(cutoff * BIN_INVLEN);  /* count extra bins around lattice */
  binDim.x = (int) ceil(lnx * h * BIN_INVLEN) + 2*c;
  binDim.y = (int) ceil(lny * h * BIN_INVLEN) + 2*c;
  binDim.z = (int) ceil(lnz * h * BIN_INVLEN) + 2*c;
  nbins = binDim.x * binDim.y * binDim.z;
  binBaseAddr = (cl_float4 *) calloc(nbins * BIN_DEPTH, sizeof(cl_float4));
  binZeroAddr = binBaseAddr + ((c * binDim.y + c) * binDim.x + c) * BIN_DEPTH;

  bincntBaseAddr = (int *) calloc(nbins, sizeof(int));
  bincntZeroAddr = bincntBaseAddr + (c * binDim.y + c) * binDim.x + c;

  /* create neighbor list */
  if (ceilf(BIN_LENGTH / (8*h)) == floorf(BIN_LENGTH / (8*h))) {
    float s = sqrtf(3);
    float r2 = (cutoff + s*BIN_LENGTH) * (cutoff + s*BIN_LENGTH);
    int cnt = 0;
    /* develop neighbor list around 1 cell */
    if (2*c + 1 > NBRLIST_DIM) {
      fprintf(stderr, "must have cutoff <= %f\n",
          (NBRLIST_DIM-1)/2 * BIN_LENGTH);
      return -1;
    }
    for (k = -c;  k <= c;  k++) {
      for (j = -c;  j <= c;  j++) {
        for (i = -c;  i <= c;  i++) {
          if ((i*i + j*j + k*k)*BIN_LENGTH*BIN_LENGTH >= r2) continue;
          nbrlist[cnt].x = i;
          nbrlist[cnt].y = j;
          nbrlist[cnt].z = k;
          cnt++;
        }
      }
    }
    nbrlistlen = cnt;
  }
  else if (8*h <= 2*BIN_LENGTH) {
    float s = 2.f*sqrtf(3);
    float r2 = (cutoff + s*BIN_LENGTH) * (cutoff + s*BIN_LENGTH);
    int cnt = 0;
    /* develop neighbor list around 3-cube of cells */
    if (2*c + 3 > NBRLIST_DIM) {
      fprintf(stderr, "must have cutoff <= %f\n",
          (NBRLIST_DIM-3)/2 * BIN_LENGTH);
      return -1;
    }
    for (k = -c;  k <= c;  k++) {
      for (j = -c;  j <= c;  j++) {
        for (i = -c;  i <= c;  i++) {
          if ((i*i + j*j + k*k)*BIN_LENGTH*BIN_LENGTH >= r2) continue;
          nbrlist[cnt].x = i;
          nbrlist[cnt].y = j;
          nbrlist[cnt].z = k;
          cnt++;
        }
      }
    }
    nbrlistlen = cnt;
  }
  else {
    fprintf(stderr, "must have h <= %f\n", 0.25 * BIN_LENGTH);
    return -1;
  }

  /* perform geometric hashing of atoms into bins */
  {
    /* array of extra atoms, permit average of one extra per bin */
    Atom *extra_atoms = (Atom *) calloc(nbins, sizeof(Atom));
    int extra_len = 0;
    
    for (n = 0;  n < natoms;  n++) {
      cl_float4 p;
      p.x = atom[n].x - xlo;
      p.y = atom[n].y - ylo;
      p.z = atom[n].z - zlo;
      p.w = atom[n].q;
      i = (int) floorf(p.x * BIN_INVLEN);
      j = (int) floorf(p.y * BIN_INVLEN);
      k = (int) floorf(p.z * BIN_INVLEN);
      if (i >= -c && i < binDim.x - c &&
	  j >= -c && j < binDim.y - c &&
	  k >= -c && k < binDim.z - c &&
	  atom[n].q != 0) {
	int index = (k * binDim.y + j) * binDim.x + i;
	cl_float4 *bin = binZeroAddr + index * BIN_DEPTH;
	int bindex = bincntZeroAddr[index];
	if (bindex < BIN_DEPTH) {
	  /* copy atom into bin and increase counter for this bin */
	  bin[bindex] = p;
	  bincntZeroAddr[index]++;
	}
	else {
	  /* add index to array of extra atoms to be computed with CPU */
	  if (extra_len >= nbins) {
	    fprintf(stderr, "exceeded space for storing extra atoms\n");
	    return -1;
	  }
	  extra_atoms[extra_len] = atom[n];
	  extra_len++;
	}
      }
      else {
	/* excluded atoms are either outside bins or neutrally charged */
	num_excluded++;
      }
    }

    /* Save result */
    extra = (Atoms *)malloc(sizeof(Atoms));
    extra->atoms = extra_atoms;
    extra->size = extra_len;
  }

  /* bin stats */
  sum = total = 0;
  for (n = 0;  n < nbins;  n++) {
    binHistoFull[ bincntBaseAddr[n] ]++;
    sum += bincntBaseAddr[n];
    total += BIN_DEPTH;
  }
  avgFillFull = sum / (float) total;
  sum = total = 0;
  for (k = 0;  k < binDim.z - 2*c;  k++) {
    for (j = 0;  j < binDim.y - 2*c;  j++) {
      for (i = 0;  i < binDim.x - 2*c;  i++) {
        int index = (k * binDim.y + j) * binDim.x + i;
        binHistoCover[ bincntZeroAddr[index] ]++;
        sum += bincntZeroAddr[index];
        total += BIN_DEPTH;
      }
    }
  }
  avgFillCover = sum / (float) total;

  if (verbose) {
    /* report */
    printf("number of atoms = %d\n", natoms);
    printf("lattice spacing = %g\n", h);
    printf("cutoff distance = %g\n", cutoff);
    printf("\n");
    printf("requested lattice dimensions = %d %d %d\n", nx, ny, nz);
    printf("requested space dimensions = %g %g %g\n", nx*h, ny*h, nz*h);
    printf("expanded lattice dimensions = %d %d %d\n", lnx, lny, lnz);
    printf("expanded space dimensions = %g %g %g\n", lnx*h, lny*h, lnz*h);
    printf("number of bytes for lattice data = %u\n", (unsigned int) (lnall*sizeof(float)));
    printf("\n");
    printf("bin padding thickness = %d\n", c);
    printf("bin cover dimensions = %d %d %d\n",
        binDim.x - 2*c, binDim.y - 2*c, binDim.z - 2*c);
    printf("bin full dimensions = %d %d %d\n", binDim.x, binDim.y, binDim.z);
    printf("number of bins = %d\n", nbins);
    printf("total number of atom slots = %d\n", nbins * BIN_DEPTH);
    printf("%% overhead space = %g\n",
        (natoms / (double) (nbins * BIN_DEPTH)) * 100);
    printf("number of bytes for bin data = %u\n",
        (unsigned int)(nbins * BIN_DEPTH * sizeof(cl_float4)));
    printf("\n");
    printf("bin histogram with padding:\n");
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      printf("     number of bins with %d atoms:  %d\n", n, binHistoFull[n]);
      sum += binHistoFull[n];
    }
    printf("     total number of bins:  %d\n", sum);
    printf("     %% average fill:  %g\n", avgFillFull * 100);
    printf("\n");
    printf("bin histogram excluding padding:\n");
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      printf("     number of bins with %d atoms:  %d\n", n, binHistoCover[n]);
      sum += binHistoCover[n];
    }
    printf("     total number of bins:  %d\n", sum);
    printf("     %% average fill:  %g\n", avgFillCover * 100);
    printf("\n");
    printf("number of extra atoms = %d\n", extra->size);
    printf("%% atoms that are extra = %g\n", (extra->size / (double) natoms) * 100);
    printf("\n");

    /* sanity check on bins */
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      sum += n * binHistoFull[n];
    }
    sum += extra->size + num_excluded;
    printf("sanity check on bin histogram with edges:  "
        "sum + others = %d\n", sum);
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      sum += n * binHistoCover[n];
    }
    sum += extra->size + num_excluded;
    printf("sanity check on bin histogram excluding edges:  "
        "sum + others = %d\n", sum);
    printf("\n");

    /* neighbor list */
    printf("neighbor list length = %d\n", nbrlistlen);
    printf("\n");
  }

  printf("Ok!\n");

  pb_Context* pb_context;
  pb_context = pb_InitOpenCLContext(parameters);
  if (pb_context == NULL) {
    fprintf (stderr, "Error: No OpenCL platform/device can be found."); 
    return -1;
  }

  printf("Ok!\n");

  cl_int clStatus;
  cl_device_id clDevice = (cl_device_id) pb_context->clDeviceId;
  cl_platform_id clPlatform = (cl_platform_id) pb_context->clPlatformId;
  cl_context clContext = (cl_context) pb_context->clContext;

  cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
  CHECK_ERROR("clCreateCommandQueue")

  pb_SetOpenCL(&clContext, &clCommandQueue);
  
  //const char* clSource[] = {readFile("src/opencl_base/kernel.cl")};
  //cl_program clProgram = clCreateProgramWithSource(clContext,1,clSource,NULL,&clStatus);
  cl_program clProgram = clCreateProgramWithBuiltInKernels(
      clContext, 1, &clDevice, "opencl_cutoff_potential_lattice", &clStatus);
  CHECK_ERROR("clCreateProgramWithSource")

  char clOptions[50];
  sprintf(clOptions,"-I src/opencl_base");  //-cl-nv-verbose

  clStatus = clBuildProgram(clProgram,1,&clDevice,clOptions,NULL,NULL);
  if (clStatus != CL_SUCCESS) {
    size_t string_size = 0;
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 
                          0, NULL, &string_size);
    char* string = (char*)malloc(string_size*sizeof(char));
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 
                          string_size, string, NULL);
    puts(string);
  }

  CHECK_ERROR("clBuildProgram")

  cl_kernel clKernel = clCreateKernel(clProgram,"opencl_cutoff_potential_lattice",&clStatus);
  CHECK_ERROR("clCreateKernel")

  /* setup OpenCL kernel parameters */
  blockDim[0] = 8;
  blockDim[1] = 8;
  blockDim[2] = 2;
  gridDim[0] = 4 * xRegionDim * blockDim[0];
  gridDim[1] = yRegionDim * blockDim[1];
  gridDim[2] = 1 * blockDim[2];

  /* allocate and initialize memory on OpenCL device */
  pb_SwitchToTimer(timers, pb_TimerID_COPY);
  if (verbose) {
    printf("Allocating %.2fMB on OpenCL device for potentials\n",
           lnall * sizeof(float) / (double) (1024*1024));
  }

  regionZeroCl = clCreateBuffer(clContext,CL_MEM_WRITE_ONLY,lnall*sizeof(float),NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")

  // clMemSet(clCommandQueue,regionZeroCl,0,lnall*sizeof(float));

  if (verbose) {
    printf("Allocating %.2fMB on OpenCL device for atom bins\n",
           nbins * BIN_DEPTH * sizeof(cl_float4) / (double) (1024*1024));
  }

  binBaseCl = clCreateBuffer(clContext,CL_MEM_READ_ONLY,nbins*BIN_DEPTH*sizeof(cl_float4),NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
 
  clStatus = clEnqueueWriteBuffer(clCommandQueue,binBaseCl,CL_TRUE,0,nbins*BIN_DEPTH*sizeof(cl_float4),binBaseAddr,0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")

  //Sub buffers are not supported in OpenCL v1.0
  int offset = ((c * binDim.y + c) * binDim.x + c) * BIN_DEPTH;  

  NbrListLen = clCreateBuffer(clContext,CL_MEM_READ_ONLY,sizeof(int),NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
  clStatus = clEnqueueWriteBuffer(clCommandQueue,NbrListLen,CL_TRUE,0,sizeof(int),&nbrlistlen,0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")

  NbrList = clCreateBuffer(clContext,CL_MEM_READ_ONLY,NBRLIST_MAXLEN*sizeof(xyz),NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
  clStatus = clEnqueueWriteBuffer(clCommandQueue,NbrList,CL_TRUE,0,nbrlistlen*sizeof(xyz),nbrlist,0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")

  if (verbose) 
    printf("\n");

  clStatus = clSetKernelArg(clKernel,0,sizeof(int),&(binDim.x));
  clStatus = clSetKernelArg(clKernel,1,sizeof(int),&(binDim.y));
  clStatus = clSetKernelArg(clKernel,2,sizeof(cl_mem),&binBaseCl);
  clStatus = clSetKernelArg(clKernel,3,sizeof(int),&offset);
  clStatus = clSetKernelArg(clKernel,4,sizeof(float),&h);
  clStatus = clSetKernelArg(clKernel,5,sizeof(float),&cutoff2);
  clStatus = clSetKernelArg(clKernel,6,sizeof(float),&inv_cutoff2);
  clStatus = clSetKernelArg(clKernel,7,sizeof(cl_mem),&regionZeroCl);
  clStatus = clSetKernelArg(clKernel,9,sizeof(cl_mem),&NbrListLen);
  clStatus = clSetKernelArg(clKernel,10,sizeof(cl_mem),&NbrList);
  CHECK_ERROR("clSetKernelArg")

  printf("Ok!!\n");


  /* loop over z-dimension, invoke OpenCL kernel for each x-y plane */
  pb_SwitchToTimer(timers, pb_TimerID_KERNEL);
  printf("Invoking OpenCL kernel on %d region planes...\n", zRegionDim);
  for (zRegionIndex = 0;  zRegionIndex < zRegionDim;  zRegionIndex++) {
    printf("  computing plane %d\r", zRegionIndex);
    fflush(stdout);

    clStatus = clSetKernelArg(clKernel,8,sizeof(int),&zRegionIndex);
    CHECK_ERROR("clSetKernelArg")

    printf("Ok**!2\n");

    clStatus = clEnqueueNDRangeKernel(clCommandQueue,clKernel,3,NULL,gridDim,blockDim,0,NULL,NULL);

    printf("Ok**!2\n");

    CHECK_ERROR("clEnqueueNDRangeKernel")

    printf("Ok**!2\n");

    clStatus = clFinish(clCommandQueue);

    printf("Ok**!2\n");

    CHECK_ERROR("clFinish")
  }

  printf("Ok++!\n");

  printf("Finished OpenCL kernel calls                        \n");

  /* copy result regions from OpenCL device */
  pb_SwitchToTimer(timers, pb_TimerID_COPY);
  clStatus = clEnqueueReadBuffer(clCommandQueue,regionZeroCl,CL_TRUE,0,lnall*sizeof(float),regionZeroAddr,0,NULL,NULL);
  CHECK_ERROR("clEnqueueReadBuffer")

  /* free OpenCL memory allocations */
  clStatus = clReleaseMemObject(regionZeroCl);
  clStatus = clReleaseMemObject(binBaseCl);
  clStatus = clReleaseMemObject(NbrListLen);
  clStatus = clReleaseMemObject(NbrList);
  CHECK_ERROR("clReleaseMemObject")

  clStatus = clReleaseKernel(clKernel);
  clStatus = clReleaseProgram(clProgram);
  clStatus = clReleaseCommandQueue(clCommandQueue);
  clStatus = clReleaseContext(clContext);

  //free((void*)clSource[0]);

  /* transpose regions back into lattice */
  pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);
  for (k = 0;  k < nz;  k++) {
    zRegionIndex = (k >> 3);
    zOffset = (k & 7);

    for (j = 0;  j < ny;  j++) {
      yRegionIndex = (j >> 3);
      yOffset = (j & 7);

      for (i = 0;  i < nx;  i++) {
        xRegionIndex = (i >> 3);
        xOffset = (i & 7);

        thisRegion = regionZeroAddr
          + ((zRegionIndex * yRegionDim + yRegionIndex) * xRegionDim
              + xRegionIndex) * REGION_SIZE;

        indexRegion = (zOffset * 8 + yOffset) * 8 + xOffset;
        index = (k * ny + j) * nx + i;

        lattice->lattice[index] = thisRegion[indexRegion];
      }
    }
  }

  /* handle extra atoms */
  if (extra->size > 0) {
    printf("computing extra atoms on CPU\n");
    if (cpu_compute_cutoff_potential_lattice(lattice, cutoff, extra)) {
      fprintf(stderr, "cpu_compute_cutoff_potential_lattice() failed "
          "for extra atoms\n");
      return -1;
    }
    printf("\n");
  }

  /* cleanup memory allocations */
  free(regionZeroAddr);
  free(binBaseAddr);
  free(bincntBaseAddr);
  free_atom(extra);

  return 0;
}
