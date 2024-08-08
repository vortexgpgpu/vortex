#ifndef __GAUSSIAN_ELIMINATION__
#define __GAUSSIAN_ELIMINATION__

#include "gaussianElim.h"

cl_context context = NULL;

int main(int argc, char *argv[]) {
  printf("enter demo main\n");
  float *a = NULL, *b = NULL, *finalVec = NULL;
  float *m = NULL;
  int size = 0;

  FILE *fp = NULL;

  // args
  char filename[100];
  int quiet = 0, timing = 0, platform = -1, device = -1;

  // parse command line
  if (parseCommandline(argc, argv, filename, &quiet, &timing, &platform,
                       &device, &size)) {
    printUsage();
    return 0;
  }

  context = cl_init_context(platform, device, quiet);

  if (size == 0) {
    fp = fopen(filename, "r");
    fscanf(fp, "%d", &size);

    printf("using %dx%d input matrix\n", size, size);

    a = (float *)malloc(size * size * sizeof(float));

    InitMat(fp, size, a, size, size);
    // printf("The input matrix a is:\n");
    // PrintMat(a, size, size, size);
    b = (float *)malloc(size * sizeof(float));

    InitAry(fp, b, size);
    // printf("The input array b is:\n");
    // PrintAry(b, size);
  } else {
    // create the input matrix
    a = (float *)malloc(size * size * sizeof(float));
    b = (float *)malloc(size * sizeof(float));

    for (int i = 0, n = size * size; i < n; ++i) {
      a[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    for (int i = 0; i < size; ++i) {
      b[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
  }

  // create the solution matrix
  m = (float *)malloc(size * size * sizeof(float));

  // create a new vector to hold the final answer
  finalVec = (float *)malloc(size * sizeof(float));

  InitPerRun(size, m);

  // begin timing

  // run kernels
  ForwardSub(context, a, b, m, size, timing);

  // end timing
  if (!quiet) {
    printf("The result of matrix m is: \n");

    PrintMat(m, size, size, size);
    printf("The result of matrix a is: \n");
    PrintMat(a, size, size, size);
    printf("The result of array b is: \n");
    PrintAry(b, size);

    BackSub(a, b, finalVec, size);
    printf("The final solution is: \n");
    PrintAry(finalVec, size);
  }

  if (fp) fclose(fp);
  free(m);
  free(a);
  free(b);
  free(finalVec);
  // OpenClGaussianElimination(context,timing);

  cl_cleanup();

  return 0;
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub(cl_context context, float *a, float *b, float *m, int size,
                int timing) {
  // 1. set up kernels
  cl_kernel fan1_kernel, fan2_kernel;
  cl_int status = 0;
  cl_program gaussianElim_program;
  cl_event writeEvent, kernelEvent, readEvent;
  float writeTime = 0, readTime = 0, kernelTime = 0;
  float writeMB = 0, readMB = 0;
  
  gaussianElim_program = cl_compileProgram((char *)"gaussianElim_kernels.cl", NULL);
  
  fan1_kernel = clCreateKernel(gaussianElim_program, "Fan1", &status);
  status = cl_errChk(status, (char *)"Error Creating Fan1 kernel", true);
  if (status)
    exit(1);

  fan2_kernel = clCreateKernel(gaussianElim_program, "Fan2", &status);
  status = cl_errChk(status, (char *)"Error Creating Fan2 kernel", true);
  if (status)
    exit(1);

  // 2. set up memory on device and send ipts data to device

  cl_mem a_dev, b_dev, m_dev;

  cl_int error = 0;

  a_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
                         sizeof(float) * size * size, NULL, &error);

  b_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL,
                         &error);

  m_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
                         sizeof(float) * size * size, NULL, &error);

  cl_command_queue command_queue = cl_getCommandQueue();

  error = clEnqueueWriteBuffer(command_queue, a_dev,
                               1, // change to 0 for nonblocking write
                               0, // offset
                               sizeof(float) * size * size, a, 0, NULL,
                               &writeEvent);

  if (timing)
    writeTime += eventTime(writeEvent, command_queue);
  clReleaseEvent(writeEvent);

  error = clEnqueueWriteBuffer(command_queue, b_dev,
                               1, // change to 0 for nonblocking write
                               0, // offset
                               sizeof(float) * size, b, 0, NULL, &writeEvent);
  if (timing)
    writeTime += eventTime(writeEvent, command_queue);
  clReleaseEvent(writeEvent);

  error = clEnqueueWriteBuffer(command_queue, 
                               m_dev,
                               1, // change to 0 for nonblocking write
                               0, // offset
                               sizeof(float) * size * size, m, 0, NULL,
                               &writeEvent);
  if (timing)
    writeTime += eventTime(writeEvent, command_queue);
  clReleaseEvent(writeEvent);
  writeMB = (float)(sizeof(float) * size * (size + size + 1) / 1e6);

  // 3. Determine block sizes
  size_t globalWorksizeFan1[1] = {size};
  size_t globalWorksizeFan2[2] = {size, size};
  size_t localWorksizeFan1[1] = {1};
  size_t localWorksizeFan2[2] = {1, 1};

  int t;
  // 4. Setup and Run kernels
  for (t = 0; t < (size - 1); t++) {
    // kernel args
    cl_int argchk;
    argchk = clSetKernelArg(fan1_kernel, 0, sizeof(cl_mem), (void *)&m_dev);
    argchk |= clSetKernelArg(fan1_kernel, 1, sizeof(cl_mem), (void *)&a_dev);
    argchk |= clSetKernelArg(fan1_kernel, 2, sizeof(cl_mem), (void *)&b_dev);
    argchk |= clSetKernelArg(fan1_kernel, 3, sizeof(int), (void *)&size);
    argchk |= clSetKernelArg(fan1_kernel, 4, sizeof(int), (void *)&t);

    cl_errChk(argchk, "ERROR in Setting Fan1 kernel args", true);

    // launch kernel
    error =
        clEnqueueNDRangeKernel(command_queue, fan1_kernel, 1, 0,
                               globalWorksizeFan1, localWorksizeFan1, 0, NULL, &kernelEvent);

    cl_errChk(error, "ERROR in Executing Fan1 Kernel", true);
    if (timing) {
      //             printf("here1a\n");
      kernelTime += eventTime(kernelEvent, command_queue);
      //             printf("here1b\n");
    }
    clReleaseEvent(kernelEvent);
    // Fan1<<<dimGrid,dimBlock>>>(m_cuda,a_cuda,Size,t);
    // cudaThreadSynchronize();

    // kernel args
    argchk = clSetKernelArg(fan2_kernel, 0, sizeof(cl_mem), (void *)&m_dev);
    argchk |= clSetKernelArg(fan2_kernel, 1, sizeof(cl_mem), (void *)&a_dev);
    argchk |= clSetKernelArg(fan2_kernel, 2, sizeof(cl_mem), (void *)&b_dev);
    argchk |= clSetKernelArg(fan2_kernel, 3, sizeof(int), (void *)&size);
    argchk |= clSetKernelArg(fan2_kernel, 4, sizeof(int), (void *)&t);

    cl_errChk(argchk, "ERROR in Setting Fan2 kernel args", true);

    // launch kernel
    error =
        clEnqueueNDRangeKernel(command_queue, fan2_kernel, 2, 0,
                               globalWorksizeFan2, localWorksizeFan2, 0, NULL, &kernelEvent);

    cl_errChk(error, "ERROR in Executing Fan1 Kernel", true);
    if (timing) {
      //             printf("here2a\n");
      kernelTime += eventTime(kernelEvent, command_queue);
      //             printf("here2b\n");
    }
    clReleaseEvent(kernelEvent);
    // Fan2<<<dimGridXY,dimBlockXY>>>(m_cuda,a_cuda,b_cuda,Size,Size-t,t);
    // cudaThreadSynchronize();
  }
  // 5. transfer data off of device
  error =
      clEnqueueReadBuffer(command_queue, a_dev,
                          1, // change to 0 for nonblocking write
                          0, // offset
                          sizeof(float) * size * size, a, 0, NULL, &readEvent);

  cl_errChk(error, "ERROR with clEnqueueReadBuffer", true);
  if (timing)
    readTime += eventTime(readEvent, command_queue);
  clReleaseEvent(readEvent);

  error = clEnqueueReadBuffer(command_queue, b_dev,
                              1, // change to 0 for nonblocking write
                              0, // offset
                              sizeof(float) * size, b, 0, NULL, &readEvent);
  cl_errChk(error, "ERROR with clEnqueueReadBuffer", true);
  if (timing)
    readTime += eventTime(readEvent, command_queue);
  clReleaseEvent(readEvent);

  error =
      clEnqueueReadBuffer(command_queue, m_dev,
                          1, // change to 0 for nonblocking write
                          0, // offset
                          sizeof(float) * size * size, m, 0, NULL, &readEvent);

  cl_errChk(error, "ERROR with clEnqueueReadBuffer", true);
  if (timing)
    readTime += eventTime(readEvent, command_queue);
  clReleaseEvent(readEvent);
  readMB = (float)(sizeof(float) * size * (size + size + 1) / 1e6);

  if (timing) {
    printf("Matrix Size\tWrite(s) [size]\t\tKernel(s)\tRead(s)  "
           "[size]\t\tTotal(s)\n");
    printf("%dx%d      \t", size, size);

    printf("%f [%.2fMB]\t", writeTime, writeMB);

    printf("%f\t", kernelTime);

    printf("%f [%.2fMB]\t", readTime, readMB);

    printf("%f\n\n", writeTime + kernelTime + readTime);
  }
  
  cl_freeMem(a_dev);
  cl_freeMem(b_dev);
  cl_freeMem(m_dev);
  cl_freeKernel(fan1_kernel);
  cl_freeKernel(fan2_kernel);
  cl_freeProgram(gaussianElim_program);
}

float eventTime(cl_event event, cl_command_queue command_queue) {
  cl_int error = 0;
  cl_ulong eventStart, eventEnd;
  clFinish(command_queue);
  error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                  sizeof(cl_ulong), &eventStart, NULL);
  cl_errChk(error, "ERROR in Event Profiling.", true);
  error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                  sizeof(cl_ulong), &eventEnd, NULL);
  cl_errChk(error, "ERROR in Event Profiling.", true);

  return (float)((eventEnd - eventStart) / 1e9);
}

int parseCommandline(int argc, char *argv[], char *filename, int *q, int *t,
                     int *p, int *d, int* s) {
  int i;
  if (argc < 2) return 1; // error
  char flag;

  for (i = 1; i < argc; i++) {
    if (argv[i][0] == '-') { // flag
      flag = argv[i][1];
      switch (flag) {
      case 'f': // matrix file
        i++;
	      strncpy(filename,argv[i],100);
	      printf("Read file from %s \n", filename);
        break;
      case 'h': // help
        return 1;
        break;
      case 'q': // quiet
        *q = 1;
        break;
      case 't': // timing
        *t = 1;
        break;
      case 'p': // platform
        i++;
        *p = atoi(argv[i]);
        break;
      case 'd': // device
        i++;
        *d = atoi(argv[i]);
        break;
      case 's': // size
        i++;
        *s = atoi(argv[i]);
        break;
      }
    }
  }
  if ((*d >= 0 && *p < 0) ||
      (*p >= 0 &&
       *d < 0)) // both p and d must be specified if either are specified
    return 1;
  return 0;
}

void printUsage() {
  printf("Gaussian Elimination Usage\n");
  printf("\n");
  printf("gaussianElimination [filename] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./gaussianElimination matrix4.txt\n");
  printf("\n");
  printf("filename     the filename that holds the matrix data\n");
  printf("\n");
  printf("-h           Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and "
         "device)\n");
  printf("-d [int]     Choose the device (must choose both platform and "
         "device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}

/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun(int size, float *m) {
  int i;
  for (i = 0; i < size * size; i++)
    *(m + i) = 0.0;
}
void BackSub(float *a, float *b, float *finalVec, int size) {
  // solve "bottom up"
  int i, j;
  for (i = 0; i < size; i++) {
    finalVec[size - i - 1] = b[size - i - 1];
    for (j = 0; j < i; j++) {
      finalVec[size - i - 1] -= *(a + size * (size - i - 1) + (size - j - 1)) *
                                finalVec[size - j - 1];
    }
    finalVec[size - i - 1] =
        finalVec[size - i - 1] / *(a + size * (size - i - 1) + (size - i - 1));
  }
}
void InitMat(FILE *fp, int size, float *ary, int nrow, int ncol) {
  int i, j;

  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++) {
      fscanf(fp, "%f", ary + size * i + j);
    }
  }
}
/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(FILE *fp, float *ary, int ary_size) {
  int i;

  for (i = 0; i < ary_size; i++) {
    fscanf(fp, "%f", &ary[i]);
  }
}
/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int size, int nrow, int ncol) {
  int i, j;

  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++) {
      printf("%8.2f ", *(ary + size * i + j));
    }
    printf("\n");
  }
  printf("\n");
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size) {
  int i;
  for (i = 0; i < ary_size; i++) {
    printf("%.2f ", ary[i]);
  }
  printf("\n\n");
}
#endif
