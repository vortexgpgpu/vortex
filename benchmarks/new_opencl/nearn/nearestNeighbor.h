#ifndef _NEARESTNEIGHBOR
#define _NEARESTNEIGHBOR

#include <iostream>
#include <vector>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

#include "clutils.h"
//#include "utils.h"

#include <algorithm>


	
#define REC_LENGTH 49 // size of a record in db

typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

float *OpenClFindNearestNeighbors(
	cl_context context,
	int numRecords,
	std::vector<LatLong> &locations,float lat,float lng,
	int timing);

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d);
#endif
