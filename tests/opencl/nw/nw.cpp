#ifdef RD_WG_SIZE_0_0
	#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
	#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE RD_WG_SIZE
#else
	#define BLOCK_SIZE 16
#endif

#define LIMIT -999

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sys/time.h>

#ifdef NV //NVIDIA
	#include <oclUtils.h>
#else 
	#include <CL/cl.h>
#endif

//global variables

int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

// local variables
static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;

static int initialize(int use_gpu)
{
	cl_int result;
	size_t size;

	// create OpenCL context
	cl_platform_id platform_id;
	if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(1,*,0) failed\n"); return -1; }
	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
	device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
	context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
	if( !context ) { printf("ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU"); return -1; }

	// get the list of GPUs
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
	num_devices = (int) (size / sizeof(cl_device_id));
	printf("num_devices = %d\n", num_devices);
	
	if( result != CL_SUCCESS || num_devices < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list = new cl_device_id[num_devices];
	if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
	if( result != CL_SUCCESS ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }

	// create command queue for the first device
	cmd_queue = clCreateCommandQueue( context, device_list[0], 0, NULL );
	if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }
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

int maximum( int a,
		 int b,
		 int c){

	int k;
	if( a <= b )
	  k = b;
	else 
	  k = a;
	if( k <=c )
	  return(c);
	else
	  return(k);
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	fprintf(stderr, "\t<file> - filename\n");
	exit(1);
}

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

int main(int argc, char **argv){

  printf("WG size of kernel = %d \n", BLOCK_SIZE);

    int max_rows, max_cols, penalty;
	char * tempchar;
	// the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc == 4)
	{
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
		tempchar = argv[3];
	}
    else{
	     usage(argc, argv);
    }
	
	if(atoi(argv[1])%16!=0){
	fprintf(stderr,"The dimension values must be a multiple of 16\n");
	exit(1);
	}

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;

	int *reference;
	int *input_itemsets;
	int *output_itemsets;
	
	reference = (int *)malloc( max_rows * max_cols * sizeof(int) );
    input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	
	srand(7);
	
	//initialization 
	for (int i = 0 ; i < max_cols; i++){
		for (int j = 0 ; j < max_rows; j++){
			input_itemsets[i*max_cols+j] = 0;
		}
	}

	for( int i=1; i< max_rows ; i++){    //initialize the cols
			input_itemsets[i*max_cols] = rand() % 10 + 1;
	}
	
    for( int j=1; j< max_cols ; j++){    //initialize the rows
			input_itemsets[j] = rand() % 10 + 1;
	}
	
	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
		reference[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}

    for( int i = 1; i< max_rows ; i++)
       input_itemsets[i*max_cols] = -i * penalty;
	for( int j = 1; j< max_cols ; j++)
       input_itemsets[j] = -j * penalty;
	
	int sourcesize = 1024*1024;
		
	char * source = (char *)calloc(sourcesize, sizeof(char)); 
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	// read the kernel core source
	char * kernel_nw1  = "nw_kernel1";
	char * kernel_nw2  = "nw_kernel2";
	FILE * fp = fopen(tempchar, "rb"); 
	if(!fp) { printf("ERROR: unable to open '%s'\n", tempchar); return -1; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);

	int nworkitems, workgroupsize = 0;
	nworkitems = BLOCK_SIZE;

	if(nworkitems < 1 || workgroupsize < 0){
		printf("ERROR: invalid or missing <num_work_items>[/<work_group_size>]\n"); 
		return -1;
	}
		// set global and local workitems
	size_t local_work[3] = { (workgroupsize>0)?workgroupsize:1, 1, 1 };
	size_t global_work[3] = { nworkitems, 1, 1 }; //nworkitems = no. of GPU threads
	
	int use_gpu = 1;
	// OpenCL initialization
	if(initialize(use_gpu)) return -1;

	// compile kernel
	cl_int err = 0;
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
	/*{ // show warnings/errors
		static char log[65536]; memset(log, 0, sizeof(log));
		cl_device_id device_id = 0;
		err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, NULL);
		clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
		if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
	}*/
	if(err != CL_SUCCESS) { printf("ERROR: clBuildProgram() => %d\n", err); return -1; }
    	
	cl_kernel kernel1;
	cl_kernel kernel2;
	kernel1 = clCreateKernel(prog, kernel_nw1, &err);  
	kernel2 = clCreateKernel(prog, kernel_nw2, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
	clReleaseProgram(prog);
	
		
	
	// creat buffers
	cl_mem input_itemsets_d;
	cl_mem output_itemsets_d;
	cl_mem reference_d;
	
	input_itemsets_d = clCreateBuffer(context, CL_MEM_READ_WRITE, max_cols * max_rows * sizeof(int), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_item_set (size:%d) => %d\n", max_cols * max_rows, err); return -1;}
	reference_d		 = clCreateBuffer(context, CL_MEM_READ_WRITE, max_cols * max_rows * sizeof(int), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer reference (size:%d) => %d\n", max_cols * max_rows, err); return -1;}
	output_itemsets_d = clCreateBuffer(context, CL_MEM_READ_WRITE, max_cols * max_rows * sizeof(int), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer output_item_set (size:%d) => %d\n", max_cols * max_rows, err); return -1;}
	
	//write buffers
	err = clEnqueueWriteBuffer(cmd_queue, input_itemsets_d, 1, 0, max_cols * max_rows * sizeof(int), input_itemsets, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer bufIn1 (size:%d) => %d\n", max_cols * max_rows, err); return -1; }
	err = clEnqueueWriteBuffer(cmd_queue, reference_d, 1, 0, max_cols * max_rows * sizeof(int), reference, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer bufIn2 (size:%d) => %d\n", max_cols * max_rows, err); return -1; }
		
	int worksize = max_cols - 1;
	printf("worksize = %d\n", worksize);
	//these two parameters are for extension use, don't worry about it.
	int offset_r = 0, offset_c = 0;
	int block_width = worksize/BLOCK_SIZE ;
	
	clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &reference_d);
	clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &input_itemsets_d);
	clSetKernelArg(kernel1, 2, sizeof(void *), (void*) &output_itemsets_d);
	clSetKernelArg(kernel1, 3, sizeof(cl_int) * (BLOCK_SIZE + 1) *(BLOCK_SIZE+1), (void*)NULL );
	clSetKernelArg(kernel1, 4, sizeof(cl_int) *  BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	clSetKernelArg(kernel1, 5, sizeof(cl_int), (void*) &max_cols);
	clSetKernelArg(kernel1, 6, sizeof(cl_int), (void*) &penalty);
	clSetKernelArg(kernel1, 8, sizeof(cl_int), (void*) &block_width);
	clSetKernelArg(kernel1, 9, sizeof(cl_int), (void*) &worksize);
	clSetKernelArg(kernel1, 10, sizeof(cl_int), (void*) &offset_r);
	clSetKernelArg(kernel1, 11, sizeof(cl_int), (void*) &offset_c);

	clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &reference_d);
	clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &input_itemsets_d);
	clSetKernelArg(kernel2, 2, sizeof(void *), (void*) &output_itemsets_d);
	clSetKernelArg(kernel2, 3, sizeof(cl_int) * (BLOCK_SIZE + 1) *(BLOCK_SIZE+1), (void*)NULL );
	clSetKernelArg(kernel2, 4, sizeof(cl_int) * BLOCK_SIZE *BLOCK_SIZE, (void*)NULL );
	clSetKernelArg(kernel2, 5, sizeof(cl_int), (void*) &max_cols);
	clSetKernelArg(kernel2, 6, sizeof(cl_int), (void*) &penalty);
	clSetKernelArg(kernel2, 8, sizeof(cl_int), (void*) &block_width);
	clSetKernelArg(kernel2, 9, sizeof(cl_int), (void*) &worksize);
	clSetKernelArg(kernel2, 10, sizeof(cl_int), (void*) &offset_r);
	clSetKernelArg(kernel2, 11, sizeof(cl_int), (void*) &offset_c);
	
	printf("Processing upper-left matrix\n");
	for( int blk = 1 ; blk <= worksize/BLOCK_SIZE ; blk++){
	
		global_work[0] = BLOCK_SIZE * blk;
		local_work[0]  = BLOCK_SIZE;
		clSetKernelArg(kernel1, 7, sizeof(cl_int), (void*) &blk);
		err = clEnqueueNDRangeKernel(cmd_queue, kernel1, 2, NULL, global_work, local_work, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }			
	}
	clFinish(cmd_queue);
	
	printf("Processing lower-right matrix\n");
	for( int blk =  worksize/BLOCK_SIZE - 1  ; blk >= 1 ; blk--){	   
		global_work[0] = BLOCK_SIZE * blk;
		local_work[0] =  BLOCK_SIZE;
		clSetKernelArg(kernel2, 7, sizeof(cl_int), (void*) &blk);
        err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 2, NULL, global_work, local_work, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: 2 clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
	}
    
    // Lingjie Zhang modified at Nov 1, 2015
    //	clFinish(cmd_queue);
    //	fflush(stdout);
	//end Lingjie Zhang modification

    err = clEnqueueReadBuffer(cmd_queue, input_itemsets_d, 1, 0, max_cols * max_rows * sizeof(int), output_itemsets, 0, 0, 0);
	clFinish(cmd_queue);

//#define TRACEBACK	
#ifdef TRACEBACK
	
	FILE *fpo = fopen("result.txt","w");
	fprintf(fpo, "print traceback value GPU:\n");
    
	for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 )
			fprintf(fpo, "%d ", output_itemsets[ i * max_cols + j]); //print the first element
		if ( i == 0 && j == 0 )
           break;
		if ( i > 0 && j > 0 ){
			nw = output_itemsets[(i - 1) * max_cols + j - 1];
		    w  = output_itemsets[ i * max_cols + j - 1 ];
            n  = output_itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
		    nw = n = LIMIT;
		    w  = output_itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
		    nw = w = LIMIT;
            n  = output_itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		//traceback = maximum(nw, w, n);
		int new_nw, new_w, new_n;
		new_nw = nw + reference[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
            traceback = n;
			
		fprintf(fpo, "%d ", traceback);

		if(traceback == nw )
		{i--; j--; continue;}

        else if(traceback == w )
		{j--; continue;}

        else if(traceback == n )
		{i--; continue;}

		else
		;
	}
	
	fclose(fpo);

#endif

	printf("Computation Done\n");
    // OpenCL shutdown
	if(shutdown()) return -1;

	clReleaseMemObject(input_itemsets_d);
	clReleaseMemObject(output_itemsets_d);
	clReleaseMemObject(reference_d);

	free(reference);
	free(input_itemsets);
	free(output_itemsets);
	
}

