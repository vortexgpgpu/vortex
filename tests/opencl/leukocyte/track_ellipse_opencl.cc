#include "find_ellipse.h"
#include "track_ellipse_opencl.h"
#include "OpenCL_helper_library.h"


// Host and device arrays to hold device pointers to input matrices
int *host_I_offsets;
cl_mem device_I_offsets;
// Host and device arrays to hold sizes of input matrices
int *host_m_array, *host_n_array;
cl_mem device_m_array, device_n_array;

// Host and device arrays to hold matrices for all cells
// (so we can copy to and from the device in a single transfer)
float *host_I_all;
cl_mem device_I_all, device_IMGVF_all;
size_t total_mem_size;

// The number of work items per work group
const size_t local_work_size = 256;

cl_bool compiled = FALSE;
cl_kernel IMGVF_kernel;

int read_kernel_file_t(const char* filename, uint8_t** data, size_t* size) {
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

// Host function that launches an OpenCL kernel to compute the MGVF matrices for the specified cells
void IMGVF_OpenCL(MAT **I, MAT **IMGVF, double vx, double vy, double e, int max_iterations, double cutoff, int num_cells) {

	cl_int error;
	
	// Initialize the data on the GPU
	IMGVF_OpenCL_init(I, num_cells);
	
	if (! compiled) {
    /*
    // Load the kernel source from the file
		const char *source = load_kernel_source("track_ellipse_kernel.cl");
		size_t sourceSize = strlen(source);

		// Compile the kernel
		cl_program program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &error);
		check_error(error, __FILE__, __LINE__);
		error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
		// Show compiler warnings/errors
		static char log[65536]; memset(log, 0, sizeof(log));
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
		if (strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
		check_error(error, __FILE__, __LINE__);
    */

    uint8_t *kernel_bin = NULL;
    size_t kernel_size;
    cl_int binary_status = 0;
    if (0 != read_kernel_file_t("track_ellipse_kernel.pocl", &kernel_bin, &kernel_size))
      std::abort();

    cl_program program = clCreateProgramWithBinary(
          context, 1, &device, &kernel_size,
          (const uint8_t**)&kernel_bin, &binary_status, &error);
    free(kernel_bin);
		check_error(error, __FILE__, __LINE__);

	  error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
		check_error(error, __FILE__, __LINE__);

		// Create the IMGVF kernels
		IMGVF_kernel = clCreateKernel(program, "IMGVF_kernel", &error);
		check_error(error, __FILE__, __LINE__);

		// Record that compiling has already completed
		compiled = TRUE;
	}
	
	// Setup execution parameters
	size_t num_work_groups = num_cells;
	size_t global_work_size = num_work_groups * local_work_size;
	
	// Convert double-precision parameters to single-precision
	float vx_float = (float) vx;
	float vy_float = (float) vy;
	float e_float = (float) e;
	float cutoff_float = (float) cutoff;
	
	// Set the kernel arguments
	clSetKernelArg(IMGVF_kernel, 0, sizeof(cl_mem), (void *) &device_IMGVF_all);
	clSetKernelArg(IMGVF_kernel, 1, sizeof(cl_mem), (void *) &device_I_all);
	clSetKernelArg(IMGVF_kernel, 2, sizeof(cl_mem), (void *) &device_I_offsets);
	clSetKernelArg(IMGVF_kernel, 3, sizeof(cl_mem), (void *) &device_m_array);
	clSetKernelArg(IMGVF_kernel, 4, sizeof(cl_mem), (void *) &device_n_array);
	clSetKernelArg(IMGVF_kernel, 5, sizeof(cl_float), (void *) &vx_float);
	clSetKernelArg(IMGVF_kernel, 6, sizeof(cl_float), (void *) &vy_float);
	clSetKernelArg(IMGVF_kernel, 7, sizeof(cl_float), (void *) &e_float);
	clSetKernelArg(IMGVF_kernel, 8, sizeof(cl_int), (void *) &max_iterations);
	clSetKernelArg(IMGVF_kernel, 9, sizeof(cl_float), (void *) &cutoff_float);
	
	// Compute the MGVF on the GPU
	error = clEnqueueNDRangeKernel(command_queue, IMGVF_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
	check_error(error, __FILE__, __LINE__);
	
	// Check for kernel errors
	error = clFinish(command_queue);
	check_error(error, __FILE__, __LINE__);
	
	// Copy back the final results from the GPU
	IMGVF_OpenCL_cleanup(IMGVF, num_cells);
}


// Initializes data on the GPU for the MGVF kernel
void IMGVF_OpenCL_init(MAT **IE, int num_cells) {
	cl_int error;
	
	// Allocate array of offsets to each cell's image
	size_t mem_size = sizeof(int) * num_cells;
	host_I_offsets = (int *) malloc(mem_size);
	device_I_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, mem_size, NULL, &error);
	check_error(error, __FILE__, __LINE__);
	
	// Allocate arrays to hold the dimensions of each cell's image
	host_m_array = (int *) malloc(mem_size);
	host_n_array = (int *) malloc(mem_size);
	device_m_array = clCreateBuffer(context, CL_MEM_READ_ONLY, mem_size, NULL, &error);
	check_error(error, __FILE__, __LINE__);
	device_n_array = clCreateBuffer(context, CL_MEM_READ_ONLY, mem_size, NULL, &error);
	check_error(error, __FILE__, __LINE__);
	
	// Figure out the size of all of the matrices combined
	int i, j, cell_num;
	size_t total_size = 0;
	for (cell_num = 0; cell_num < num_cells; cell_num++) {
		MAT *I = IE[cell_num];
		size_t size = I->m * I->n;
		total_size += size;
	}
	total_mem_size = total_size * sizeof(float);
	
	// Allocate host memory just once for all cells
	host_I_all = (float *) malloc(total_mem_size);
	
	// Allocate device memory just once for all cells
	device_I_all = clCreateBuffer(context, CL_MEM_READ_ONLY, total_mem_size, NULL, &error);
	check_error(error, __FILE__, __LINE__);
	device_IMGVF_all = clCreateBuffer(context, CL_MEM_READ_WRITE, total_mem_size, NULL, &error);
	check_error(error, __FILE__, __LINE__);
	
	// Copy each initial matrix into the allocated host memory
	int offset = 0;
	for (cell_num = 0; cell_num < num_cells; cell_num++) {
		MAT *I = IE[cell_num];
		
		// Determine the size of the matrix
		int m = I->m, n = I->n;
		int size = m * n;
		
		// Store memory dimensions
		host_m_array[cell_num] = m;
		host_n_array[cell_num] = n;
		
		// Store offsets to this cell's image
		host_I_offsets[cell_num] = offset;
		
		// Copy matrix I (which is also the initial IMGVF matrix) into the overall array
		for (i = 0; i < m; i++)
			for (j = 0; j < n; j++)
				host_I_all[offset + (i * n) + j] = (float) m_get_val(I, i, j);
		
		offset += size;
	}
	
	// Copy I matrices (which are also the initial IMGVF matrices) to device
	error = clEnqueueWriteBuffer(command_queue, device_I_all, CL_TRUE, 0, total_mem_size, (void *) host_I_all, 0, NULL, NULL);
	check_error(error, __FILE__, __LINE__);
	error = clEnqueueWriteBuffer(command_queue, device_IMGVF_all, CL_TRUE, 0, total_mem_size, (void *) host_I_all, 0, NULL, NULL);
	check_error(error, __FILE__, __LINE__);
	
	// Copy offsets array to device
	error = clEnqueueWriteBuffer(command_queue, device_I_offsets, CL_TRUE, 0, mem_size, (void *) host_I_offsets, 0, NULL, NULL);
	check_error(error, __FILE__, __LINE__);
	
	// Copy memory dimension arrays to device
	error = clEnqueueWriteBuffer(command_queue, device_m_array, CL_TRUE, 0, mem_size, (void *) host_m_array, 0, NULL, NULL);
	check_error(error, __FILE__, __LINE__);
	error = clEnqueueWriteBuffer(command_queue, device_n_array, CL_TRUE, 0, mem_size, (void *) host_n_array, 0, NULL, NULL);
	check_error(error, __FILE__, __LINE__);
}


// Copies the results of the MGVF kernel back to the host
void IMGVF_OpenCL_cleanup(MAT **IMGVF_out_array, int num_cells) {
	cl_int error;
	
	// Copy the result matrices from the device to the host
	float *host_IMGVF_all = (cl_float *) clEnqueueMapBuffer(command_queue, device_IMGVF_all, CL_TRUE, CL_MAP_READ, 0, total_mem_size, 0, NULL, NULL, &error);
	check_error(error, __FILE__, __LINE__);
	
	// Copy each result matrix into its appropriate host matrix
	int cell_num, offset = 0;	
	for (cell_num = 0; cell_num < num_cells; cell_num++) {
		MAT *IMGVF_out = IMGVF_out_array[cell_num];
		
		// Determine the size of the matrix
		int m = IMGVF_out->m, n = IMGVF_out->n, i, j;
		// Pack the result into the matrix
		for (i = 0; i < m; i++)
			for (j = 0; j < n; j++)
				m_set_val(IMGVF_out, i, j, (double) host_IMGVF_all[offset + (i * n) + j]);
		
		offset += (m * n);
	}
	
	// Unmap results buffer
	error = clEnqueueUnmapMemObject(command_queue, device_IMGVF_all, (void *) host_IMGVF_all, 0, NULL, NULL);
	check_error(error, __FILE__, __LINE__);
	
	// Free device memory
	clReleaseMemObject(device_m_array);
	clReleaseMemObject(device_n_array);
	clReleaseMemObject(device_IMGVF_all);
	clReleaseMemObject(device_I_all);
	clReleaseMemObject(device_I_offsets);
	
	// Free host memory
	free(host_m_array);
	free(host_n_array);
	free(host_I_all);
	free(host_I_offsets);
}
