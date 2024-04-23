
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include "bucketsort.h"
#include "mergesort.h"
#include <time.h>
/* #define VERIFY Y */
/* #define TIMER Y */

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define SIZE (1000)
//#define SIZE (1000000)

#define DATA_SIZE (1024)
#define MAX_SOURCE_SIZE (0x100000)
#define HISTOGRAM_SIZE (1024 * sizeof(unsigned int))

#ifdef TIMING
#include "timing.h"

struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_init_start, tv_init_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

int platform_id_inuse = 0; //platform id used (default : 0)
int device_id_inuse = 0; //deviced id used (default : 0)

////////////////////////////////////////////////////////////////////////////////
int compare(const void *a, const void *b) {
	if(*((float *)a) < *((float *)b)) return -1;
	else if(*((float *)a) > *((float *)b)) return 1;
	else return 0;
}

////////////////////////////////////////////////////////////////////////////////
cl_float4*runMergeSort(int listsize, int divisions,
                               cl_float4 *d_origList, cl_float4 *d_resultList,
                               int *sizes, int *nullElements,
                       unsigned int *origOffsets);

//---------------------------------------
//Read command line parameters
//
void clCmdParams(int argc, char* argv[]) {
	for (int i =0; i < argc; ++i) {
		switch (argv[i][1]) {
		  case 'd':	 //--d stands for device id used in computaion
			if (++i < argc)
				sscanf(argv[i], "%u", &device_id_inuse);
			else {
				printf("Could not read argument after option %s\n", argv[i-1]);
				exit(0);
			}
			break;
		  case 'p':	 //--p stands for platform id used in computaion
			if (++i < argc)
				sscanf(argv[i], "%u", &platform_id_inuse);
			else {
				printf("Could not read argument after option %s\n", argv[i-1]);
				exit(0);
			}
			break;
		default:
			;
		}
	}
}

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
    
    unsigned int correct;               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    unsigned int *results;

    // Fill our data set with random float values
    //
    clCmdParams(argc, argv);
    
    int numElements = 0 ;
        
    if(strcmp(argv[1],"r") ==0) {
        numElements = SIZE;
	}
    else {
		FILE *fp;
        fp = fopen(argv[1],"r");
        if(fp == NULL) {
            printf("Error reading file \n");
            exit(EXIT_FAILURE);
        }
        int count = 0;
        float c;
        
        while(fscanf(fp,"%f",&c) != EOF) {
            count++;
        }
        fclose(fp);
        
        numElements = count;
    }
    printf("Sorting list of %d floats.\n", numElements);
    int mem_size = (numElements + (DIVISIONS*4))*sizeof(float);
	// Allocate enough for the input list
	float *cpu_idata = (float *)malloc(mem_size);
    float *cpu_odata = (float *)malloc(mem_size);
	// Allocate enough for the output list on the cpu side
	float *d_output = (float *)malloc(mem_size);
	// Allocate enough memory for the output list on the gpu side
	float *gpu_odata = (float *)malloc(mem_size);
	float datamin = FLT_MAX;
	float datamax = -FLT_MAX;
    
    if(strcmp(argv[1],"r")==0) {
    for (int i = 0; i < numElements; i++) {
        // Generate random floats between 0 and 1 for the input data
		cpu_idata[i] = ((float) rand() / RAND_MAX);
  
        //Compare data at index to data minimum, if less than current minimum, set that element as new minimum
		datamin = fminf(cpu_idata[i], datamin);
        //Same as above but for maximum
		datamax = fmaxf(cpu_idata[i], datamax);
	}
    }
    else {
        FILE *fp;
        fp = fopen(argv[1],"r");
        for(int i = 0; i < numElements; i++) {
            fscanf(fp,"%f",&cpu_idata[i]);
            datamin = fminf(cpu_idata[i], datamin);
            datamax = fmaxf(cpu_idata[i],datamax);
        }
	}
    FILE *tp;
    const char filename2[]="./hybridinput.txt";
    tp = fopen(filename2,"w");
    for(int i = 0; i < SIZE; i++) {
        fprintf(tp,"%f ",cpu_idata[i]);
    }

    fclose(tp);
    memcpy(cpu_odata, cpu_idata, mem_size);

    /* bucketsort */

    clock_t gpu_start = clock();
#ifdef  TIMING
    gettimeofday(&tv_total_start, NULL);
#endif
    init_bucketsort(numElements);
    int *sizes = (int*) malloc(DIVISIONS * sizeof(int));
    int *nullElements = (int*) malloc(DIVISIONS * sizeof(int));
    unsigned int *origOffsets = (unsigned int *) malloc((DIVISIONS + 1) * sizeof(int));
    clock_t bucketsort_start = clock();
    bucketSort(cpu_idata,d_output,numElements,sizes,nullElements,datamin,datamax, origOffsets);
    clock_t bucketsort_diff = clock() - bucketsort_start;
#ifdef  TIMING
	gettimeofday(&tv_close_start, NULL);
#endif
    finish_bucketsort();
#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif
    double bucketTime = getBucketTime();

    cl_float4 *d_origList = (cl_float4*) d_output;
    cl_float4 *d_resultList = (cl_float4*) cpu_idata;

    int newlistsize = 0;
    for(int i = 0; i < DIVISIONS; i++){
        newlistsize += sizes[i] * 4;
    }

    /* mergesort */

#ifdef  TIMING
    gettimeofday(&tv_init_start, NULL);
#endif
    init_mergesort(newlistsize);
#ifdef  TIMING
	gettimeofday(&tv_init_end, NULL);
	tvsub(&tv_init_end, &tv_init_start, &tv);
	init_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif
    clock_t mergesort_start = clock();
    cl_float4 *mergeresult = runMergeSort(newlistsize,DIVISIONS,d_origList,d_resultList,sizes,nullElements,origOffsets);
    clock_t mergesort_diff = clock() - mergesort_start;
#ifdef  TIMING
	gettimeofday(&tv_close_start, NULL);
#endif
    finish_mergesort();
#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif
    gpu_odata = (float*)mergeresult;
#ifdef TIMER
    clock_t gpu_diff = clock() - gpu_start;
    int gpu_msec = gpu_diff * 1000 / CLOCKS_PER_SEC;
    int bucketsort_msec = bucketsort_diff * 1000 / CLOCKS_PER_SEC;
    int mergesort_msec = mergesort_diff * 1000 / CLOCKS_PER_SEC;
    double mergeTime = getMergeTime();

    printf("GPU execution time: %0.3f ms  \n", bucketsort_msec+mergesort_msec+bucketTime+mergeTime);
    printf("  --Bucketsort execution time: %0.3f ms \n", bucketsort_msec+bucketTime);
    printf("  --Mergesort execution time: %0.3f ms \n", mergesort_msec+mergeTime);
#endif

#ifdef  TIMING
	gettimeofday(&tv_total_end, NULL);
	tvsub(&tv_total_end, &tv_total_start, &tv);
	total_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	printf("Init: %f\n", init_time);
	printf("MemAlloc: %f\n", mem_alloc_time);
	printf("HtoD: %f\n", h2d_time);
	printf("Exec: %f\n", kernel_time);
	printf("DtoH: %f\n", d2h_time);
	printf("Close: %f\n", close_time);
	printf("Total: %f\n", total_time);
#endif

#ifdef VERIFY
    clock_t cpu_start = clock(), cpu_diff;
    
    qsort(cpu_odata, numElements, sizeof(float), compare);
    cpu_diff = clock() - cpu_start;
    int cpu_msec = cpu_diff * 1000 / CLOCKS_PER_SEC;
    printf("CPU execution time: %d ms  \n", cpu_msec);
    printf("Checking result...");
    
	// Result checking
	int count = 0;
	for(int i = 0; i < numElements; i++){
		if(cpu_odata[i] != gpu_odata[i])
		{
			printf("Sort missmatch on element %d: \n", i);
			printf("CPU = %f : GPU = %f\n", cpu_odata[i], gpu_odata[i]);
			count++;
			break;
		}
    }
	if(count == 0) printf("PASSED.\n");
	else printf("FAILED.\n");
#endif
    
#ifdef OUTPUT
    FILE *tp1;
    const char filename3[]="./hybridoutput.txt";
    tp1 = fopen(filename3,"w");
    for(int i = 0; i < SIZE; i++) {
        fprintf(tp1,"%f ",cpu_idata[i]);
    }
    
    fclose(tp1);
#endif
    

//    printf("%d \n",cpu_odata[1]);
//    int summy = 0;
//    for(int i =0; i < HISTOGRAM_SIZE; i++)
//        summy+=cpu_odata[i];
//    printf("%d \n", summy);
    return 0;
}



