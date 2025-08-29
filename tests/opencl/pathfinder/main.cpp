/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include "OpenCL.h"
#include "timing.h"

using namespace std;

// halo width along one direction when advancing to the next iteration
#define HALO     1
#define STR_SIZE 256
#define DEVICE   0
#define M_SEED   9
// #define BENCH_PRINT
#define IN_RANGE(x, min, max)	((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

// Program variables.
int   rows = -1, cols = -1;
int   Ne = rows * cols;
int*  grid_data;
int** wall;
int*  result;
int   pyramid_height = -1;
int   verbose = 0;

// OCL config
int platform_id_inuse = 0;            // platform id in use (default: 0)
int device_id_inuse = 0;              //device id in use (default : 0)

#define TIMING

#ifdef TIMING
	struct timeval tv;
	struct timeval tv_total_start, tv_total_end;
	struct timeval tv_init_end;
	struct timeval tv_h2d_start, tv_h2d_end;
	struct timeval tv_d2h_start, tv_d2h_end;
	struct timeval tv_kernel_start, tv_kernel_end;
	struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
	struct timeval tv_close_start, tv_close_end;
	float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
		  d2h_time = 0, close_time = 0, total_time = 0;
#endif

void init(int argc, char** argv)
{
    int cur_arg;
	for (cur_arg = 1; cur_arg<argc; cur_arg++) {
        if (strcmp(argv[cur_arg], "-c") == 0) {
            if (argc >= cur_arg + 1) {
                cols = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "-r") == 0) {
            if (argc >= cur_arg + 1) {
                rows = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "-h") == 0) {
            if (argc >= cur_arg + 1) {
                pyramid_height = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        if (strcmp(argv[cur_arg], "-v") == 0) {
			verbose = 1;
        }
        else if (strcmp(argv[cur_arg], "-p") == 0) {
            if (argc >= cur_arg + 1) {
                platform_id_inuse = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "-d") == 0) {
            if (argc >= cur_arg + 1) {
                device_id_inuse = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
    }

	if (cols < 0 || rows < 0 || pyramid_height < 0)
	{
        fprintf(stderr, "usage: %s <-r rows> <-c cols> <-h pyramid_height> [-v] [-p platform_id] [-d device_id] [-t device_type]\n", argv[0]);
		exit(0);
	}
	grid_data = new int[rows * cols];
	wall = new int*[rows];
	for (int n = 0; n < rows; n++)
	{
		// wall[n] is set to be the nth row of the data array.
		wall[n] = grid_data + cols * n;
	}
	result = new int[cols];

	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			wall[i][j] = rand() % 10;
		}
	}
#ifdef BENCH_PRINT
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			printf("%d ", wall[i][j]);
		}
		printf("\n");
	}
#endif
}

void fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);
}

int main(int argc, char** argv)
{
	init(argc, argv);
	
	// Pyramid parameters.
	int borderCols = (pyramid_height) * HALO;
	// int smallBlockCol = ?????? - (pyramid_height) * HALO * 2;
	// int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

	
	/* printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
	   pyramid_height, cols, borderCols, NUMBER_THREADS, blockCols, smallBlockCol); */

	int size = rows * cols;

	// Create and initialize the OpenCL object.
	OpenCL cl(verbose);  // 1 means to display output (debugging mode).

#ifdef  TIMING
    gettimeofday(&tv_total_start, NULL);
#endif
	cl.init();    // 1 means to use GPU. 0 means use CPU.

	cl.gwSize(rows * cols);

	// Create and build the kernel.
	string kn = "dynproc_kernel";  // the kernel name, for future use.
	cl.createKernel(kn);

#ifdef  TIMING
	gettimeofday(&tv_init_end, NULL);
	tvsub(&tv_init_end, &tv_total_start, &tv);
	init_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    cl_int* h_outputBuffer = (cl_int*)malloc(16384 * sizeof(cl_int));
    for (int i = 0; i < 16384; i++) {
        h_outputBuffer[i] = 0;
	}

#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_start, NULL);
#endif
	// Allocate device memory.
    cl_mem d_gpuWall = clCreateBuffer(cl.ctxt(), CL_MEM_READ_ONLY,
        sizeof(cl_int) * (size - cols), NULL, NULL);

    cl_mem d_gpuResult[2];

    d_gpuResult[0] = clCreateBuffer(cl.ctxt(), CL_MEM_READ_WRITE,
        sizeof(cl_int) * cols, NULL, NULL);

    d_gpuResult[1] = clCreateBuffer(cl.ctxt(), CL_MEM_READ_WRITE,
        sizeof(cl_int) * cols, NULL, NULL);

    cl_mem d_outputBuffer = clCreateBuffer(cl.ctxt(),
        CL_MEM_READ_WRITE, sizeof(cl_int) * 16384, NULL, NULL);

#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_end, NULL);
    tvsub(&tv_mem_alloc_end, &tv_mem_alloc_start, &tv);
    mem_alloc_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    cl_event write_event[3];
    clEnqueueWriteBuffer(cl.q(), d_gpuWall, 1, 0,
        sizeof(cl_int) * (size - cols), (grid_data + cols), 0, 0, &write_event[0]);

    clEnqueueWriteBuffer(cl.q(), d_gpuResult[0], 1, 0,
        sizeof(cl_int) * cols, grid_data, 0, 0, &write_event[1]);

    clEnqueueWriteBuffer(cl.q(), d_outputBuffer, 1, 0,
        sizeof(cl_int) * 16384, h_outputBuffer, 0, 0, &write_event[2]);

#ifdef TIMING
    h2d_time += probe_event_time(write_event[0], cl.q());
    h2d_time += probe_event_time(write_event[1], cl.q());
    h2d_time += probe_event_time(write_event[2], cl.q());
#endif

	int src = 1, final_ret = 0;
	for (int t = 0; t < rows - 1; t += pyramid_height)
	{
		int temp = src;
		src = final_ret;
		final_ret = temp;

		printf("Processing row %d to %d\n", t, t + pyramid_height - 1);
		printf("cl local size: %zu\n", cl.localSize());

		// Calculate this for the kernel argument...
		int arg0 = MIN(pyramid_height, rows-t-1);
		int theHalo = HALO;

		// Set the kernel arguments.
		clSetKernelArg(cl.kernel(kn), 0,  sizeof(cl_int), (void*) &arg0);
		clSetKernelArg(cl.kernel(kn), 1,  sizeof(cl_mem), (void*) &d_gpuWall);
		clSetKernelArg(cl.kernel(kn), 2,  sizeof(cl_mem), (void*) &d_gpuResult[src]);
		clSetKernelArg(cl.kernel(kn), 3,  sizeof(cl_mem), (void*) &d_gpuResult[final_ret]);
		clSetKernelArg(cl.kernel(kn), 4,  sizeof(cl_int), (void*) &cols);
		clSetKernelArg(cl.kernel(kn), 5,  sizeof(cl_int), (void*) &rows);
		clSetKernelArg(cl.kernel(kn), 6,  sizeof(cl_int), (void*) &t);
		clSetKernelArg(cl.kernel(kn), 7,  sizeof(cl_int), (void*) &borderCols);
		clSetKernelArg(cl.kernel(kn), 8,  sizeof(cl_int), (void*) &theHalo);
		clSetKernelArg(cl.kernel(kn), 9,  sizeof(cl_int) * (cl.localSize()), 0);
		clSetKernelArg(cl.kernel(kn), 10, sizeof(cl_int) * (cl.localSize()), 0);		
		clSetKernelArg(cl.kernel(kn), 11, sizeof(cl_mem), (void*) &d_outputBuffer);
		cl.launch(kn);
	}

	// Copy results back to host.
	cl_event event;
	clEnqueueReadBuffer(cl.q(),                   // The command queue.
	                    d_gpuResult[final_ret],   // The result on the device.
	                    CL_TRUE,                  // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                        // Offset. None in this case.
	                    sizeof(cl_int)*cols,      // Size to copy.
	                    result,                   // The pointer to the memory on the host.
	                    0,                        // Number of events in wait list. Not used.
	                    NULL,                     // Event wait list. Not used.
	                    &event);                  // Event object for determining status. Not used.
#ifdef TIMING
    d2h_time += probe_event_time(event,cl.q());
#endif
    clReleaseEvent(event);

	// Copy string buffer used for debugging from device to host.
	clEnqueueReadBuffer(cl.q(),                   // The command queue.
	                    d_outputBuffer,           // Debug buffer on the device.
	                    CL_TRUE,                  // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                        // Offset. None in this case.
	                    sizeof(cl_char)*16384,    // Size to copy.
	                    h_outputBuffer,           // The pointer to the memory on the host.
	                    0,                        // Number of events in wait list. Not used.
	                    NULL,                     // Event wait list. Not used.
	                    &event);                  // Event object for determining status. Not used.
#ifdef TIMING
    d2h_time += probe_event_time(event,cl.q());
#endif

	// Tack a null terminator at the end of the string.
	h_outputBuffer[16383] = '\0';
	
#ifdef BENCH_PRINT
	for (int i = 0; i < cols; i++)
		printf("%d ", grid_data[i]);
	printf("\n");
	for (int i = 0; i < cols; i++)
		printf("%d ", result[i]);
	printf("\n");
#endif

#ifdef  TIMING
	gettimeofday(&tv_close_start, NULL);
#endif
	clReleaseMemObject(d_gpuWall);
	clReleaseMemObject(d_gpuResult[0]);
	clReleaseMemObject(d_gpuResult[1]);
	clReleaseMemObject(d_outputBuffer);
#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	// Memory cleanup here.
	delete[] grid_data;
	delete[] wall;
	delete[] result;

	return EXIT_SUCCESS;
}
