/***********************************************
	streamcluster.h
	: header file to streamcluster
	
	- original code from PARSEC Benchmark Suite
	- parallelization with CUDA API has been applied by
	
	Sang-Ha (a.k.a Shawn) Lee - sl4ge@virginia.edu
	University of Virginia
	Department of Electrical and Computer Engineering
	Department of Computer Science
	
	:revised by
	Jianbin Fang - j.fang@tudelft.nl
	Delft University of Technology
	Faculty of Electrical Engineering, Mathematics and Computer Science
	Department of Software Technology
	Parallel and Distributed Systems Group
	on 15/03/2011
***********************************************/

#ifndef STREAMCLUSTER_H
#define STREAMCLUSTER_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/resource.h>
#include <limits.h>


#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

using namespace std;

/* this structure represents a point */
/* these will be passed around to avoid copying coordinates */
typedef struct {
  float weight;
  float * coord;
  long assign;  /* number of point where this one is assigned */
  float cost;  /* cost of that assignment, weight*distance */
} Point;

/* this is the array of points */
typedef struct {
  long num; /* number of points; may not be N if this is a sample */
  int dim;  /* dimensionality */
  Point *p; /* the array itself */
} Points;

struct pkmedian_arg_t
{
  Points* points;
  long kmin;
  long kmax;
  long* kfinal;
  int pid;
  pthread_barrier_t* barrier;
};

class PStream {
public:
  virtual size_t read( float* dest, int dim, int num ) = 0;
  virtual int ferror() = 0;
  virtual int feof() = 0;
  virtual ~PStream() {
  }
};

//synthetic stream
class SimStream : public PStream {
public:
  SimStream(long n_ ) {
    n = n_;
  }
  size_t read( float* dest, int dim, int num ) {
    size_t count = 0;
    for( int i = 0; i < num && n > 0; i++ ) {
      for( int k = 0; k < dim; k++ ) {
	dest[i*dim + k] = lrand48()/(float)INT_MAX;
      }
      n--;
      count++;
    }
    return count;
  }
  int ferror() {
    return 0;
  }
  int feof() {
    return n <= 0;
  }
  ~SimStream() { 
  }
private:
  long n;
};

class FileStream : public PStream {
public:
  FileStream(char* filename) {
    fp = fopen( filename, "rb");
    if( fp == NULL ) {
      fprintf(stderr,"error opening file %s\n.",filename);
      exit(1);
    }
  }
  size_t read( float* dest, int dim, int num ) {
    return std::fread(dest, sizeof(float)*dim, num, fp); 
  }
  int ferror() {
    return std::ferror(fp);
  }
  int feof() {
    return std::feof(fp);
  }
  ~FileStream() {
    printf("closing file stream\n");
    fclose(fp);
  }
private:
  FILE* fp;
};

/* function prototypes */
double gettime();
int isIdentical(float*, float*, int);
//static int floatcomp(const void*, const void*);
void shuffle(Points*);
void intshuffle(int*, int);
float waste(float);
float dist(Point, Point, int);
float pspeedy(Points*, float, long, int, pthread_barrier_t*);
float pgain_old(long, Points*, float, long int*, int, pthread_barrier_t*);
float pFL(Points*, int*, int, float, long*, float, long, float, int, pthread_barrier_t*);
int selectfeasible_fast(Points*, int**, int, int, pthread_barrier_t*);
float pkmedian(Points*, long, long, long*, int, pthread_barrier_t*);
int contcenters(Points*);
void copycenters(Points*, Points*, long*, long);
void* localSearchSub(void*);
void localSearch(Points*, long, long, long*);
void outcenterIDs(Points*, long*, char*);
void streamCluster(PStream*, long, long, int, long, long, char*);
float pgain(long, Points*, float, long int*, int, bool*, int*, bool*, double*, double*, double*, double*, double*);
void allocDevMem(int, int, int);
void freeDevMem();
void quit(char*);

#endif
