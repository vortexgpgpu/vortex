//--by Jianbin Fang

#define __CL_ENABLE_EXCEPTIONS
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cstring>
#include <CL/cl.h>
#include <math.h>
//#include "CLHelper.h"
#include "util.h"

#define MAX_THREADS_PER_BLOCK 256//256
int work_group_size = 512;//512

// OpenCL variables
static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;
uint8_t *kernel_bin = NULL;
cl_int status;
int shutdown()
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

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
	 shutdown();			                                                     \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
	   shutdown();			                                                   \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
   })

//Structure to hold a node information
struct Node {
    int starting;
    int no_of_edges;
};

int initialize(int use_gpu);
int shutdown();
void dump2file(int *adjmatrix, int num_nodes);
void print_vector(int *vector, int num);
void print_vectorf(float *vector, int num);

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
    return -1;
  }
  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);
  
  fclose(fp);
  
  return 0;
}


//----------------------------------------------------------
//--bfs on cpu
//--programmer:	jianbin
//--date:	26/01/2011
//--note: width is changed to the new_width
//----------------------------------------------------------
void run_bfs_cpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
                 int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, \
                 char *h_graph_visited, int *h_cost_ref)
{
    char stop;
    int k = 0;
    do {
        //if no thread changes this value then the loop stops
        stop=false;
        for(int tid = 0; tid < no_of_nodes; tid++ ) {
            if (h_graph_mask[tid] == true) {
                h_graph_mask[tid]=false;
                for(int i=h_graph_nodes[tid].starting; i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++) {
                    int id = h_graph_edges[i];	//--cambine: node id is connected with node tid
                    if(!h_graph_visited[id]) {	//--cambine: if node id has not been visited, enter the body below
                        h_cost_ref[id]=h_cost_ref[tid]+1;
                        h_updating_graph_mask[id]=true;
                    }
                }
            }
        }

        for(int tid=0; tid< no_of_nodes ; tid++ ) {
            if (h_updating_graph_mask[tid] == true) {
                h_graph_mask[tid]=true;
                h_graph_visited[tid]=true;
                stop=true;
                h_updating_graph_mask[tid]=false;
            }
        }
        k++;
    } while(stop);
}

int main(int argc, char * argv[])
{
    int no_of_nodes;
    int edge_list_size;
    FILE *fp;
    Node *h_graph_nodes;
    char *h_graph_mask, *h_updating_graph_mask, *h_graph_visited;

	
	    char *input_f = "graph4096.txt";
	    printf("Reading File\n");
	    // Read in Graph from a file
	    fp = fopen(input_f, "r");
    if (!fp) {
      printf("Error Reading graph file\n");
      return 0;
    }

    printf("Reading File completed!\n");

	 int source = 0;
     fscanf(fp,"%d",&no_of_nodes);
int num_of_blocks = 1;
    int num_of_threads_per_block = no_of_nodes;

    // Make execution Parameters according to the number of nodes
    // Distribute threads across multiple Blocks if necessary
    if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
      num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK);
      num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
    }
    work_group_size = num_of_threads_per_block;
	printf("Before allocation\n");
	
	//1. Allocate and initialize host memory  
	h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
		h_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
		h_updating_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
		h_graph_visited = (char*) malloc(sizeof(char)*no_of_nodes);
        int start, edgeno;
        // initalize the memory
        for(int i = 0; i < no_of_nodes; i++) {
            fscanf(fp,"%d %d",&start,&edgeno);
            h_graph_nodes[i].starting = start;
            h_graph_nodes[i].no_of_edges = edgeno;
            h_graph_mask[i]=false;
            h_updating_graph_mask[i]=false;
            h_graph_visited[i]=false;
        }printf("initialzefa mem \n");
        //read the source node from the file
        fscanf(fp,"%d",&source);
        source=0;
        //set the source node as true in the mask
        h_graph_mask[source]=true;
        h_graph_visited[source]=true;
        fscanf(fp,"%d",&edge_list_size);
        int id,cost;
        int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
        for(int i=0; i < edge_list_size ; i++) {
            fscanf(fp,"%d",&id);
            fscanf(fp,"%d",&cost);
            h_graph_edges[i] = id;
        }
        if(fp)
            fclose(fp);
    // allocate mem for the result on host side
		 int	*h_cost = (int*) malloc(sizeof(int)*no_of_nodes);
        int *h_cost_ref = (int*)malloc(sizeof(int)*no_of_nodes);
        for(int i=0; i<no_of_nodes; i++) {
            h_cost[i]=-1;
            h_cost_ref[i] = -1;
        }
        h_cost[source]=0;
        h_cost_ref[source]=0;
	printf("1. \n");
	
	//2. Get platform and device id + setup 
	  cl_platform_id platform_id;
	  cl_device_id device_id = NULL;
	  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
	  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
	 printf("2. \n");

	
	//3. Create context
	  //cl_context context = NULL;
	  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));
 printf("3. \n");

	//4. Create command queue
	 cmd_queue = CL_CHECK2(clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &_err));  
 printf("4. \n");

	//5. Create memory buffers
	 //create device side buffers
	//cl_status status;
	 char h_over;
    cl_mem d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, d_over;
 d_graph_nodes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, no_of_nodes*sizeof(Node), NULL, &status);
        d_graph_edges = clCreateBuffer(context, CL_MEM_WRITE_ONLY, edge_list_size*sizeof(int), NULL, &status);
        d_graph_mask = clCreateBuffer(context, CL_MEM_READ_WRITE, no_of_nodes*sizeof(char), NULL, &status);
        d_updating_graph_mask =  clCreateBuffer(context, CL_MEM_READ_WRITE, no_of_nodes*sizeof(char), NULL, &status);
        d_graph_visited =  clCreateBuffer(context, CL_MEM_READ_WRITE, no_of_nodes*sizeof(char), NULL, &status);
        d_cost =  clCreateBuffer(context, CL_MEM_READ_WRITE, no_of_nodes*sizeof(int), NULL, &status);
        d_over =  clCreateBuffer(context, CL_MEM_READ_WRITE, no_of_nodes*sizeof(char), NULL, &status);

 printf("5. \n");

	//6. Copy host variables to device
    cl_event event;
    /*status = clEnqueueWriteBuffer(cmd_queue, d_mem, CL_TRUE, 0, size, h_mem_ptr, 0, NULL, &event);*/
	status = clEnqueueWriteBuffer(cmd_queue, d_graph_nodes, CL_TRUE, 0, no_of_nodes*sizeof(Node), h_graph_nodes, 0, NULL, &event);
	status = clEnqueueWriteBuffer(cmd_queue, d_graph_edges, CL_TRUE, 0, edge_list_size*sizeof(int),  h_graph_edges, 0, NULL, &event);
	status = clEnqueueWriteBuffer(cmd_queue, d_graph_mask, CL_TRUE, 0, no_of_nodes*sizeof(char),  h_graph_mask, 0, NULL, &event);
	status = clEnqueueWriteBuffer(cmd_queue, d_updating_graph_mask, CL_TRUE, 0, no_of_nodes*sizeof(char), h_updating_graph_mask, 0, NULL, &event);
	status = clEnqueueWriteBuffer(cmd_queue, d_graph_visited, CL_TRUE, 0, no_of_nodes*sizeof(char), h_graph_visited, 0, NULL, &event);
	 status = clEnqueueWriteBuffer(cmd_queue, d_cost, CL_TRUE, 0, no_of_nodes*sizeof(int), h_cost, 0, NULL, &event);
 printf("6. \n");

	//7. Use the kernel function and read kernel, check if success
	// read kernel binary from file  
	size_t kernel_size;
	if (0 != read_kernel_file("kernel.pocl", &kernel_bin, &kernel_size)){
	    return -1;}
 printf("7. \n");

	//8. Create program from kernel (with binary)
		printf("Create program from kernel source\n");
		cl_int binary_status;
		 cl_program prog = NULL;
		prog = CL_CHECK2(clCreateProgramWithBinary(context, 1, &device_id, &kernel_size, (const uint8_t**)&kernel_bin, &binary_status, &_err));
	if (prog == NULL) {
	    shutdown();
	    return -1;
  }
 printf("8. \n");

	//9. Build program and create opencl kernel
	  // Build program
	  CL_CHECK(clBuildProgram(prog, 1, &device_id, NULL, NULL, NULL));
	 //create GPU kernels	
    cl_kernel kernel1, kernel2;
	 char * kernelpr1  = "BFS_1";
     char * kernelpr2  = "BFS_2";
	kernel1 = clCreateKernel(prog, kernelpr1, &status);  
    if(status != CL_SUCCESS) { printf("ERROR: clCreateKernel() 1 => %d\n", status); return -1; }
    kernel2 = clCreateKernel(prog, kernelpr2, &status);  
    if(status != CL_SUCCESS) { printf("ERROR: clCreateKernel() 2 => %d\n", status); return -1; }

   clReleaseProgram(prog);
 printf("9. \n");

  //10. Declare global and local worksize
/*        int num_of_blocks = 1;
        int num_of_threads_per_block = no_of_nodes;
        if(no_of_nodes>MAX_THREADS_PER_BLOCK) {
            num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK);
            num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
        }
        work_group_size = num_of_threads_per_block;*/
 printf("10. \n");
	//Variables for execution time /*CHANGE 2*/
	cl_event myEvent; 
	cl_ulong startTime, endTime; 
	cl_ulong totaltime=0, temptime;
int iter = 0;
	//11. Set kernel args and execute opencl kernel 
do {

		h_over = false;
status = clEnqueueWriteBuffer(cmd_queue, d_over,
            CL_TRUE, 0, sizeof(char), &h_over, 0, NULL, 0);
	if(status != CL_SUCCESS) { fprintf(stderr, "ERROR: write h_over variable (%d)\n", status); return -1; }

		//11. a. Global and local work size
	int work_items = no_of_nodes;
	if(work_items%work_group_size != 0)	{
        work_items = work_items + (work_group_size-(work_items%work_group_size));}
    size_t local_work_size[2] = {work_group_size, 1};
    size_t global_work_size[2] = {work_items, 1};
            printf("11 a. worksize set \n");
 
       //11. b. a. Set args kernel 1
 clSetKernelArg(kernel1, 0, sizeof(d_graph_nodes), &d_graph_nodes);
 clSetKernelArg(kernel1, 1, sizeof(d_graph_edges), &d_graph_edges);
 clSetKernelArg(kernel1, 2, sizeof(d_graph_mask), &d_graph_mask);
 clSetKernelArg(kernel1, 3, sizeof(d_updating_graph_mask), &d_updating_graph_mask);
 clSetKernelArg(kernel1, 4, sizeof(d_graph_visited), &d_graph_visited);
 clSetKernelArg(kernel1, 5, sizeof(d_cost), &d_cost);
 clSetKernelArg(kernel1, 6, sizeof(int), &no_of_nodes);
 printf("11 b a. Set args kernel 1 \n");

	//11. b. b. Invoke kernel 1
status = clEnqueueNDRangeKernel(cmd_queue, kernel1, 1, NULL, global_work_size, local_work_size, 0, 0, &myEvent);	
if(status != CL_SUCCESS) { fprintf(stderr, "ERROR: kernel1 (%d)\n", status); return -1; }
 printf("11 b b. Invoke kernel 1 \n");
clGetEventProfilingInfo(myEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
clGetEventProfilingInfo(myEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
temptime = endTime - startTime;
printf("Time in iter %d is %lu\n", iter, temptime);
totaltime = totaltime + temptime;
    //11. c. a Set args kernel 2
 clSetKernelArg(kernel2, 0, sizeof(d_graph_mask), &d_graph_mask);
 clSetKernelArg(kernel2, 1, sizeof(d_updating_graph_mask), &d_updating_graph_mask);
 clSetKernelArg(kernel2, 2, sizeof(d_graph_visited), &d_graph_visited);
clSetKernelArg(kernel2, 3, sizeof(d_over), &d_over);
clSetKernelArg(kernel2, 4, sizeof(int), &no_of_nodes);
 printf("11 c a. Set args kernel 2 \n");

        //11. c. b. Invoke kernel 2
status = clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work_size, local_work_size, 0, 0, &myEvent);	
    if(status != CL_SUCCESS) { fprintf(stderr, "ERROR: kernel1 (%d)\n", status); return -1; }
 printf("11 c b. Invoke kernel 2 \n");
clGetEventProfilingInfo(myEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
clGetEventProfilingInfo(myEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
temptime = endTime - startTime;
totaltime = totaltime + temptime;

		//11. d. Copy the termination variable back
		 status = clEnqueueReadBuffer(cmd_queue, d_over, 1, 0, sizeof(char), &h_over, 0, 0, 0);
        if(status != CL_SUCCESS) { fprintf(stderr, "ERROR: read stop_d variable (%d)\n", status); return -1; }
 printf("11 d. Copy back h over \n");

printf("iteration number %d\n", iter);
iter++;
} while(h_over);

clFinish(cmd_queue);
 printf("11. \n");

//clGetEventProfilingInfo(myEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL); 
//clGetEventProfilingInfo(myEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL); 
unsigned long kernelExecTime = totaltime;
printf("Execution time : %lu\n", kernelExecTime);


status = clEnqueueReadBuffer(cmd_queue, d_cost, CL_TRUE, 0, no_of_nodes*sizeof(int), h_cost, 0,0, 0);

    if(status != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueReadBuffer()=>%d failed\n", status); return -1; }

 if(d_graph_nodes!= NULL) clReleaseMemObject(d_graph_nodes);
 if(d_graph_edges!= NULL) clReleaseMemObject(d_graph_edges);
 if(d_graph_mask!= NULL) clReleaseMemObject(d_graph_mask);
 if(d_updating_graph_mask!= NULL) clReleaseMemObject(d_updating_graph_mask);
if(d_cost!= NULL) clReleaseMemObject(d_cost);
if(d_over!= NULL) clReleaseMemObject(d_over);
 shutdown();
       
        //--CPU 
 printf("cpu \n");

        // initalize the memory again
        for(int i = 0; i < no_of_nodes; i++) {
            h_graph_mask[i]=false;
            h_updating_graph_mask[i]=false;
            h_graph_visited[i]=false;
        }
        //set the source node as true in the mask
        source=0;
        h_graph_mask[source]=true;
        h_graph_visited[source]=true;
        run_bfs_cpu(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost_ref);
        //---------------------------------------------------------
        //--result verification
        compare_results<int>(h_cost_ref, h_cost, no_of_nodes);
        //release host memory
        free(h_graph_nodes);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);

     /*catch(std::string msg) {
        std::cout<<"--cambine: exception in main ->"<<msg<<std::endl;
        //release host memory
        free(h_graph_nodes);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
    }*/

    return 0;
}

