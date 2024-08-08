#ifndef __NEAREST_NEIGHBOR__
#define __NEAREST_NEIGHBOR__

#include "nearestNeighbor.h"

cl_context context = NULL;

int main(int argc, char *argv[]) {
  std::vector<Record> records;
  float *recordDistances;
  // LatLong locations[REC_WINDOW];
  std::vector<LatLong> locations;
  int i;
  // args
  char filename[100];
  int resultsCount = 5, quiet = 0, timing = 0, platform = -1, device = -1;
  float lat = 30, lng = 90;

  // parse command line
  if (parseCommandline(argc, argv, filename, &resultsCount, &lat, &lng, &quiet,
                       &timing, &platform, &device)) {
    printUsage();
    return 0;
  }

  int numRecords = loadData(filename, records, locations);

  // for(i=0;i<numRecords;i++)
  //    printf("%s, %f,
  //    %f\n",(records[i].recString),locations[i].lat,locations[i].lng);

  printf("Number of records: %d\n", numRecords);
  printf("Finding the %d closest neighbors.\n", resultsCount);

  if (resultsCount > numRecords)
    resultsCount = numRecords;

  context = cl_init_context(platform, device, quiet);
  
  recordDistances = OpenClFindNearestNeighbors(context, numRecords, locations,
                                               lat, lng, timing);

  // find the resultsCount least distances
  findLowest(records, recordDistances, numRecords, resultsCount);

  // print out results  
  if (!quiet) {
    for (i = 0; i < resultsCount; i++) {
      printf("%s --> Distance=%f\n", records[i].recString, records[i].distance);
    }
  }
  
  // verify result
  int errors = 0;
  for (i = 1; i < resultsCount; ++i) {
    if (records[i].distance < records[i-1].distance) {
      ++errors;
    }
  }
  
  free(recordDistances);

  cl_cleanup();

  if (errors != 0) {
    printf("Failed!\n");
  } else {
    printf("Passed!\n");
  }
  
  return errors;
}

float *OpenClFindNearestNeighbors(cl_context context, int numRecords,
                                  std::vector<LatLong> &locations, float lat,
                                  float lng, int timing) {
  cl_int status;
  
  // 1. set up kernel
  cl_kernel NN_kernel;  
  cl_program cl_NN_program;
  cl_NN_program = cl_compileProgram((char *)"nearestNeighbor_kernel.cl", NULL);

  NN_kernel = clCreateKernel(cl_NN_program, "NearestNeighbor", &status);
  cl_errChk(status, (char *)"Error Creating Nearest Neighbor kernel", true);
  
  // 2. set up memory on device and send ipts data to device
  // copy ipts(1,2) to device
  // also need to alloate memory for the distancePoints
  cl_mem d_locations;
  cl_mem d_distances;

  cl_int error = 0;

  d_locations = clCreateBuffer(context, CL_MEM_READ_ONLY,
                               sizeof(LatLong) * numRecords, NULL, &error);
  cl_errChk(error, "ERROR: clCreateBuffer() failed", true);

  d_distances = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               sizeof(float) * numRecords, NULL, &error);
  cl_errChk(error, "ERROR: clCreateBuffer() failed", true);  

  cl_command_queue command_queue = cl_getCommandQueue();
  cl_event writeEvent, kernelEvent, readEvent;
  error = clEnqueueWriteBuffer(command_queue, d_locations,
                               1, // change to 0 for nonblocking write
                               0, // offset
                               sizeof(LatLong) * numRecords, &locations[0], 0,
                               NULL, &writeEvent);
  cl_errChk(error, "ERROR: clEnqueueWriteBuffer() failed", true);

  // 3. send arguments to device
  cl_int argchk;
  argchk = clSetKernelArg(NN_kernel, 0, sizeof(cl_mem), (void *)&d_locations);
  argchk |= clSetKernelArg(NN_kernel, 1, sizeof(cl_mem), (void *)&d_distances);
  argchk |= clSetKernelArg(NN_kernel, 2, sizeof(int), (void *)&numRecords);
  argchk |= clSetKernelArg(NN_kernel, 3, sizeof(float), (void *)&lat);
  argchk |= clSetKernelArg(NN_kernel, 4, sizeof(float), (void *)&lng);

  cl_errChk(argchk, "ERROR in Setting Nearest Neighbor kernel args", true);

  // 4. enqueue kernel
  size_t globalWorkSize[1];
  size_t localWorkSize[1] = {1};
  globalWorkSize[0] = numRecords;
  if (numRecords % 64)
    globalWorkSize[0] += 64 - (numRecords % 64);
  // printf("Global Work Size: %zu\n",globalWorkSize[0]);

  error = clEnqueueNDRangeKernel(command_queue, NN_kernel, 1, 0, globalWorkSize,
                                 localWorkSize, 0, NULL, &kernelEvent);

  cl_errChk(error, "ERROR in Executing Kernel NearestNeighbor", true);

  // 5. transfer data off of device

  // create distances std::vector
  float *distances = (float *)malloc(sizeof(float) * numRecords);

  error = clEnqueueReadBuffer(command_queue, d_distances,
                              1, // change to 0 for nonblocking write
                              0, // offset
                              sizeof(float) * numRecords, distances, 0, NULL,
                              &readEvent);

  cl_errChk(error, "ERROR with clEnqueueReadBuffer", true);

  clFinish(command_queue);

  if (timing) {    
    cl_ulong eventStart, eventEnd, totalTime = 0;
    printf("# Records\tWrite(s) [size]\t\tKernel(s)\tRead(s)  "
           "[size]\t\tTotal(s)\n");
    printf("%d        \t", numRecords);
    // Write Buffer
    error = clGetEventProfilingInfo(writeEvent, CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong), &eventStart, NULL);
    cl_errChk(error, "ERROR in Event Profiling (Write Start)", true);
    error = clGetEventProfilingInfo(writeEvent, CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong), &eventEnd, NULL);
    cl_errChk(error, "ERROR in Event Profiling (Write End)", true);

    printf("%f [%.2fMB]\t", (float)((eventEnd - eventStart) / 1e9),
           (float)((sizeof(LatLong) * numRecords) / 1e6));
    totalTime += eventEnd - eventStart;
    // Kernel
    error = clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong), &eventStart, NULL);
    cl_errChk(error, "ERROR in Event Profiling (Kernel Start)", true);
    error = clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong), &eventEnd, NULL);
    cl_errChk(error, "ERROR in Event Profiling (Kernel End)", true);

    printf("%f\t", (float)((eventEnd - eventStart) / 1e9));
    totalTime += eventEnd - eventStart;
    // Read Buffer
    error = clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong), &eventStart, NULL);
    cl_errChk(error, "ERROR in Event Profiling (Read Start)", true);
    error = clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong), &eventEnd, NULL);
    cl_errChk(error, "ERROR in Event Profiling (Read End)", true);

    printf("%f [%.2fMB]\t", (float)((eventEnd - eventStart) / 1e9),
           (float)((sizeof(float) * numRecords) / 1e6));
    totalTime += eventEnd - eventStart;

    printf("%f\n\n", (float)(totalTime / 1e9));
  }

  // 6. return finalized data and release buffers
  clReleaseEvent(writeEvent);  
  clReleaseEvent(kernelEvent);
  clReleaseEvent(readEvent);  
  cl_freeMem(d_locations);
  cl_freeMem(d_distances);
  cl_freeKernel(NN_kernel);
  cl_freeProgram(cl_NN_program);

  return distances;
}

int loadData(char *filename, std::vector<Record> &records,
             std::vector<LatLong> &locations) {
  FILE *flist, *fp;
  int i = 0;
  char dbname[64];
  int recNum = 0;

  /**Main processing **/

  int q = 0;

  flist = fopen(filename, "r");
  while (!feof(flist)) {
    /**
    * Read in REC_WINDOW records of length REC_LENGTH
    * If this is the last file in the filelist, then done
    * else open next file to be read next iteration
    */       
    if (fscanf(flist, "%s\n", dbname) != 1) {
      printf("error reading filelist\n");
      exit(0);
    }
    printf("loading db: %s\n", dbname);
    fp = fopen(dbname, "r");
    if (!fp) {
      printf("error opening a db\n");
      exit(1);
    }    
    // read each record
    while (!feof(fp)) {
      Record record;
      LatLong latLong;
      fgets(record.recString, 49, fp);
      fgetc(fp); // newline
      if (feof(fp))
        break;

      // parse for lat and long
      char substr[6];

      for (i = 0; i < 5; i++)
        substr[i] = *(record.recString + i + 28);
      substr[5] = '\0';
      latLong.lat = atof(substr);

      for (i = 0; i < 5; i++)
        substr[i] = *(record.recString + i + 33);
      substr[5] = '\0';
      latLong.lng = atof(substr);

      locations.push_back(latLong);
      records.push_back(record);
      recNum++;
      /*if (0 == (recNum % 500))
        break;*/
    }
    
    /*if (++q == 3)
        break;*/
    fclose(fp);
  }
  fclose(flist);
  return recNum;
}

void findLowest(std::vector<Record> &records, float *distances, int numRecords,
                int topN) {
  int i, j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;

  for (i = 0; i < topN; i++) {
    minLoc = i;
    for (j = i; j < numRecords; j++) {
      val = distances[j];
      if (val < distances[minLoc])
        minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char *filename, int *r, float *lat,
                     float *lng, int *q, int *t, int *p, int *d) {
  int i;
  if (argc < 2) return 1; // error
  strncpy(filename,argv[1],100);
  char flag;

  for (i = 1; i < argc; i++) {
    if (argv[i][0] == '-') { // flag
      flag = argv[i][1];
      switch (flag) {
      case 'r': // number of results
        i++;
        *r = atoi(argv[i]);
        break;
      case 'l':                  // lat or lng
        if (argv[i][2] == 'a') { // lat
          *lat = atof(argv[i + 1]);
        } else { // lng
          *lng = atof(argv[i + 1]);
        }
        i++;
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
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] "
         "[-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
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

#endif
