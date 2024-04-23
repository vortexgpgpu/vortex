#ifndef _C_UTIL_
#define _C_UTIL_
#include <math.h>
#include <iostream>
#include <omp.h>
#include <sys/time.h>

#ifdef RD_WG_SIZE_0_0
	#define BLOCK_SIZE_0 RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
	#define BLOCK_SIZE_0 RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_0 RD_WG_SIZE
#else
	#define BLOCK_SIZE_0 192
#endif

#ifdef RD_WG_SIZE_1_0
	#define BLOCK_SIZE_1 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
	#define BLOCK_SIZE_1 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_1 RD_WG_SIZE
#else
	#define BLOCK_SIZE_1 192
#endif

#ifdef RD_WG_SIZE_2_0
	#define BLOCK_SIZE_2 RD_WG_SIZE_2_0
#elif defined(RD_WG_SIZE_1)
	#define BLOCK_SIZE_2 RD_WG_SIZE_2
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_2 RD_WG_SIZE
#else
	#define BLOCK_SIZE_2 192
#endif

#ifdef RD_WG_SIZE_3_0
	#define BLOCK_SIZE_3 RD_WG_SIZE_3_0
#elif defined(RD_WG_SIZE_3)
	#define BLOCK_SIZE_3 RD_WG_SIZE_3
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_3 RD_WG_SIZE
#else
	#define BLOCK_SIZE_3 192
#endif

#ifdef RD_WG_SIZE_4_0
	#define BLOCK_SIZE_4 RD_WG_SIZE_4_0
#elif defined(RD_WG_SIZE_4)
	#define BLOCK_SIZE_4 RD_WG_SIZE_4
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_4 RD_WG_SIZE
#else
	#define BLOCK_SIZE_4 192
#endif


using std::endl;

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}
//-------------------------------------------------------------------
//--initialize array with maximum limit
//-------------------------------------------------------------------
template<typename datatype>
void fill(datatype *A, const int n, const datatype maxi){
    for (int j = 0; j < n; j++){
        A[j] = ((datatype) maxi * (rand() / (RAND_MAX + 1.0f)));
    }
}

//--print matrix
template<typename datatype>
void print_matrix(datatype *A, int height, int width){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			int idx = i*width + j;
			std::cout<<A[idx]<<" ";
		}
		std::cout<<std::endl;
	}

	return;
}
//-------------------------------------------------------------------
//--verify results
//-------------------------------------------------------------------
#define MAX_RELATIVE_ERROR  .002
template<typename datatype>
void verify_array(const datatype *cpuResults, const datatype *gpuResults, const int size){

    bool passed = true; 
#pragma omp parallel for
    for (int i=0; i<size; i++){
      if (fabs(cpuResults[i] - gpuResults[i]) / cpuResults[i] > MAX_RELATIVE_ERROR){
         passed = false; 
      }
    }
    if (passed){
        std::cout << "--cambine:passed:-)" << std::endl;
    }
    else{
        std::cout << "--cambine: failed:-(" << std::endl;
    }
    return ;
}
template<typename datatype>
void compare_results(const datatype *cpu_results, const datatype *gpu_results, const int size){

    bool passed = true; 
//#pragma omp parallel for
    for (int i=0; i<size; i++){
      if (cpu_results[i]!=gpu_results[i]){
         passed = false; 
      }
    }
    if (passed){
        std::cout << "--cambine:passed:-)" << std::endl;
    }
    else{
        std::cout << "--cambine: failed:-(" << std::endl;
    }
    return ;
}

#endif

