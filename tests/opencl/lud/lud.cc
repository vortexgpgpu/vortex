/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"
#include <sys/time.h>
#include <CL/cl.h>
#include "timing.h"

#include <string.h>
#include <string>
#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif

static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_uint          num_devices;
static cl_uint          num_platforms;

int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp_ = fopen(filename, "r");
  if (NULL == fp_) {
    fprintf(stderr, "Failed to load kernel. %s\n", filename);
    return -1;
  }
  fseek(fp_ , 0 , SEEK_END);
  long fsize = ftell(fp_);
  rewind(fp_);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp_);
  
  fclose(fp_);
  
  return 0;
}

static int initialize(int platform_id_inuse, int device_id_inuse)
{
    // get OpenCL platforms
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(0,0,*) failed\n"); return -1; }
	cl_platform_id all_platform_id[num_platforms];
	if (clGetPlatformIDs(num_platforms, all_platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(*,*,0) failed\n"); return -1; }
    cl_platform_id platform_id = all_platform_id[platform_id_inuse];

    // get device
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices) != CL_SUCCESS) { printf("ERROR: clGetDeviceIDs failed\n"); return -1; };
	printf("num_devices = %d\n", num_devices);
    if (device_id_inuse > (int)num_devices) {
        printf("Invalid Device Number\n");
        return -1;
    }
	device_list = new cl_device_id[num_devices];
	if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, device_list, NULL) != CL_SUCCESS) { printf("ERROR: clGetDeviceIDs failed\n"); return -1; };

    // get device type
    if (clGetDeviceInfo(device_list[device_id_inuse], CL_DEVICE_TYPE, sizeof(device_type), (void *)&device_type, NULL)!= CL_SUCCESS) { printf("ERROR: clGetDeviceIDs failed\n"); return -1; };

	// create OpenCL context
	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
	context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
	if( !context ) { printf("ERROR: clCreateContextFromType(%s) failed\n", device_type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU"); return -1; }
	printf("Create %s context\n", device_type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU");

	// create command queue for the first device
#ifdef TIMING
        cmd_queue = clCreateCommandQueue( context, device_list[device_id_inuse], CL_QUEUE_PROFILING_ENABLE, NULL );
#else
        cmd_queue = clCreateCommandQueue( context, device_list[device_id_inuse], 0, NULL );
#endif
	if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }

  printf("check %d\n", device_id_inuse);
	return 0;
}

static int shutdown()
{
	// release resources
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
	if( context ) clReleaseContext( context );
	if( device_list ) delete device_list;

	// reset all variables
	cmd_queue = 0;
	context = 0;
	device_list = 0;
	num_devices = 0;
	device_type = 0;

	return 0;
}

static int do_verify = 0;
void lud_cuda(float *d_m, int matrix_dim);

static struct option long_options[] = {
      /* name, has_arg, flag, val */
      {"input", 1, NULL, 'i'},
      {"size", 1, NULL, 's'},
      {"verify", 0, NULL, 'v'},
      {"platform", 1, NULL, 'p'},
      {"device", 1, NULL, 'd'},
      {0,0,0,0}
};

int
main ( int argc, char *argv[] )
{
    //Primitives for timing
#ifdef TIMING
    struct timeval tv;
    struct timeval tv_total_start;
    struct timeval tv_init_end;
    struct timeval tv_mem_alloc_end;
    struct timeval tv_close_start, tv_close_end;
    float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
          d2h_time = 0, close_time = 0, total_time = 0;
#endif

    printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
	size_t matrix_dim = 32; /* default matrix_dim */
	int opt, option_index=0;
	func_ret_t ret;
	const char *input_file = NULL;
	float *m, *mm;
	stopwatch sw;
	int platform_id_inuse = 0;
	int device_id_inuse = 0;

	while ((opt = getopt_long(argc, argv, "::vs:i:p:d:", 
                            long_options, &option_index)) != -1 ) {
		switch(opt){
			case 'i':
			input_file = optarg;
			break;
			case 'v':
			do_verify = 1;
			break;
        case 's':
			matrix_dim = atoi(optarg);
			printf("Generate input matrix internally, size =%lu\n", matrix_dim);
			// fprintf(stderr, "Currently not supported, use -i instead\n");
			// fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
			// exit(EXIT_FAILURE);
			break;
        case 'p':
			platform_id_inuse = atoi(optarg);
            break;
        case 'd':
			device_id_inuse = atoi(optarg);
            break;
        case '?':
			fprintf(stderr, "invalid option\n");
			break;
        case ':':
			fprintf(stderr, "missing argument\n");
			break;
        default:
			fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
                  argv[0]);
			exit(EXIT_FAILURE);
		}
	}
  
	if ( (optind < argc) || (optind == 1)) {
		fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
		exit(EXIT_FAILURE);
	}	

	if (input_file) {
		printf("Reading matrix from file %s\n", input_file);
		ret = create_matrix_from_file(&m, input_file, &matrix_dim);
		if (ret != RET_SUCCESS) {
			m = NULL;
			fprintf(stderr, "error create matrix from file %s\n", input_file);
			exit(EXIT_FAILURE);
		}
	} 
	
	else if (matrix_dim) {
	  printf("Creating matrix internally size=%lu\n", matrix_dim);
	  ret = create_matrix(&m, matrix_dim);
	  if (ret != RET_SUCCESS) {
	    m = NULL;
	    fprintf(stderr, "error create matrix internally size=%lu\n", matrix_dim);
	    exit(EXIT_FAILURE);
	  }
	}

	else {
	  printf("No input file specified!\n");
	  exit(EXIT_FAILURE);
	}

	if (do_verify){
		printf("Before LUD\n");
		// print_matrix(m, matrix_dim);
		matrix_duplicate(m, &mm, matrix_dim);
	}
	
	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char)); 
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	const char * kernel_lud_diag   = "lud_diagonal";
	const char * kernel_lud_peri   = "lud_perimeter";
	const char * kernel_lud_inter  = "lud_internal";
  /*
  FILE * fp = fopen("./lud_kernel.cl", "rb"); 
	if(!fp) { printf("ERROR: unable to open '%s'\n", "./lud_kernel.cl"); return -1; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);
*/
	// OpenCL initialization
#ifdef  TIMING
    gettimeofday(&tv_total_start, NULL);
#endif
	if(initialize(platform_id_inuse, device_id_inuse)) return -1;

	// compile kernel
	cl_int err = 0;
  /*
	const char * slist[2] = { source, 0 };
  cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithSource() => %d\n", err); return -1; }
	char clOptions[110];
	//  sprintf(clOptions,"-I../../src"); 
	sprintf(clOptions," ");
#ifdef BLOCK_SIZE
	sprintf(clOptions + strlen(clOptions), " -DBLOCK_SIZE=%d", BLOCK_SIZE);
#endif

	err = clBuildProgram(prog, 0, NULL, clOptions, NULL, NULL);
	{ // show warnings/errors
		//static char log[65536]; memset(log, 0, sizeof(log));
		//cl_device_id device_id = 0;
		//err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, NULL);
		//clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
		//if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
	}
	if(err != CL_SUCCESS) { printf("ERROR: clBuildProgram() => %d\n", err); return -1; }
  */


  uint8_t *kernel_bin = NULL;
    size_t kernel_size;
    cl_int binary_status = 0;
    if (0 != read_kernel_file("kernel.pocl", &kernel_bin, &kernel_size))
      std::abort();

    cl_program prog = clCreateProgramWithBinary(
          context, 1, &device_list[device_id_inuse], &kernel_size,
          (const uint8_t**)&kernel_bin, &binary_status, &err);
    free(kernel_bin);
	  if(err != CL_SUCCESS) { 
        printf("ERROR: program error => %d %d\n", err, device_id_inuse); return -1; }

	  err = clBuildProgram(prog, 1, &device_list[device_id_inuse], NULL, NULL, NULL);
	  if(err != CL_SUCCESS) { 
        printf("ERROR: clBuildProgram() => %d\n", err); return -1; }


	cl_kernel diagnal;
	cl_kernel perimeter;
	cl_kernel internal;
	diagnal   = clCreateKernel(prog, kernel_lud_diag, &err);  
	perimeter = clCreateKernel(prog, kernel_lud_peri, &err);  
	internal  = clCreateKernel(prog, kernel_lud_inter, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
	clReleaseProgram(prog);

#ifdef  TIMING
    gettimeofday(&tv_init_end, NULL);
    tvsub(&tv_init_end, &tv_total_start, &tv);
    init_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	//size_t local_work[3] = { 1, 1, 1 };
	//size_t global_work[3] = {1, 1, 1 }; 

	cl_mem d_m;
	d_m = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_dim*matrix_dim * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_m (size:%lu) => %d\n", matrix_dim*matrix_dim, err); return -1;} 
#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_end, NULL);
    tvsub(&tv_mem_alloc_end, &tv_init_end, &tv);
    mem_alloc_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	/* beginning of timing point */
	stopwatch_start(&sw);
	cl_event event;
	err = clEnqueueWriteBuffer(cmd_queue, d_m, 1, 0, matrix_dim*matrix_dim*sizeof(float), m, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_m (size:%lu) => %d\n", matrix_dim*matrix_dim, err); return -1; }
#ifdef TIMING
    h2d_time += probe_event_time(event,cmd_queue);
#endif
    clReleaseEvent(event);
	
	size_t i=0;
	for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
	 
	  clSetKernelArg(diagnal, 0, sizeof(void *), (void*) &d_m);
	  clSetKernelArg(diagnal, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	  clSetKernelArg(diagnal, 2, sizeof(cl_int), (void*) &matrix_dim);
	  clSetKernelArg(diagnal, 3, sizeof(cl_int), (void*) &i);
      
	  size_t global_work1[3]  = {BLOCK_SIZE, 1, 1};
	  size_t local_work1[3]  = {BLOCK_SIZE, 1, 1};
	   
	  err = clEnqueueNDRangeKernel(cmd_queue, diagnal, 2, NULL, global_work1, local_work1, 0, 0, &event);
	  if(err != CL_SUCCESS) { printf("ERROR:  diagnal clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }	
#ifdef TIMING
      kernel_time += probe_event_time(event,cmd_queue);
#endif
      clReleaseEvent(event);
	  
	  clSetKernelArg(perimeter, 0, sizeof(void *), (void*) &d_m);
	  clSetKernelArg(perimeter, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	  clSetKernelArg(perimeter, 2, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	  clSetKernelArg(perimeter, 3, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	  clSetKernelArg(perimeter, 4, sizeof(cl_int), (void*) &matrix_dim);
	  clSetKernelArg(perimeter, 5, sizeof(cl_int), (void*) &i);
	  
	  size_t global_work2[3] = {BLOCK_SIZE * 2 * ((matrix_dim-i)/BLOCK_SIZE-1), 1, 1};
	  size_t local_work2[3]  = {BLOCK_SIZE * 2, 1, 1};
      // Intel OCL does not support 0 size
      if (global_work2[0] > 0) {
          err = clEnqueueNDRangeKernel(cmd_queue, perimeter, 2, NULL, global_work2, local_work2, 0, 0, &event);
          if(err != CL_SUCCESS) { printf("ERROR:  perimeter clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }	
#ifdef TIMING
          kernel_time += probe_event_time(event,cmd_queue);
#endif
          clReleaseEvent(event);
      }
	  
	  clSetKernelArg(internal, 0, sizeof(void *), (void*) &d_m);
	  clSetKernelArg(internal, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	  clSetKernelArg(internal, 2, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	  clSetKernelArg(internal, 3, sizeof(cl_int), (void*) &matrix_dim);
	  clSetKernelArg(internal, 4, sizeof(cl_int), (void*) &i);
      
	  size_t global_work3[3] = {BLOCK_SIZE * ((matrix_dim-i)/BLOCK_SIZE-1), BLOCK_SIZE * ((matrix_dim-i)/BLOCK_SIZE-1), 1};
	  size_t local_work3[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};
      // Intel OCL does not support 0 size
      if (global_work3[0] > 0) {
          err = clEnqueueNDRangeKernel(cmd_queue, internal, 2, NULL, global_work3, local_work3, 0, 0, &event);
          if(err != CL_SUCCESS) { printf("ERROR:  internal clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }	
#ifdef TIMING
          kernel_time += probe_event_time(event,cmd_queue);
#endif
          clReleaseEvent(event);
      }
	}
	clSetKernelArg(diagnal, 0, sizeof(void *), (void*) &d_m);
	clSetKernelArg(diagnal, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	clSetKernelArg(diagnal, 2, sizeof(cl_int), (void*) &matrix_dim);
	clSetKernelArg(diagnal, 3, sizeof(cl_int), (void*) &i);
      
	size_t global_work1[3]  = {BLOCK_SIZE, 1, 1};
	size_t local_work1[3]  = {BLOCK_SIZE, 1, 1};
	err = clEnqueueNDRangeKernel(cmd_queue, diagnal, 2, NULL, global_work1, local_work1, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR:  diagnal clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }	
#ifdef TIMING
    kernel_time += probe_event_time(event,cmd_queue);
#endif
    clReleaseEvent(event);

	clFinish(cmd_queue);
	err = clEnqueueReadBuffer(cmd_queue, d_m, 1, 0, matrix_dim*matrix_dim*sizeof(float), m, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueReadBuffer  d_m (size:%lu) => %d\n", matrix_dim*matrix_dim, err); return -1; }
#ifdef TIMING
    d2h_time += probe_event_time(event,cmd_queue);
#endif
    clReleaseEvent(event);

	clFinish(cmd_queue);

#ifdef  TIMING
    gettimeofday(&tv_close_start, NULL);
#endif
	clReleaseMemObject(d_m);
#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
	tvsub(&tv_close_end, &tv_total_start, &tv);
	total_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	printf("Init: %f\n", init_time);
	printf("MemAlloc: %f\n", mem_alloc_time);
	printf("HtoD: %f\n", h2d_time);
	printf("Exec: %f\n", kernel_time);
	printf("DtoH: %f\n", d2h_time);
	printf("Close: %f\n", close_time);
	printf("Total: %f\n", total_time);
#endif

	if (do_verify){
		printf("After LUD\n");
		// print_matrix(m, matrix_dim);
		printf(">>>Verify<<<<\n");
		lud_verify(mm, m, matrix_dim); 
		free(mm);
	}

	free(m);
	
	if(shutdown()) return -1;
	
}				

/* ----------  end of function main  ---------- */


