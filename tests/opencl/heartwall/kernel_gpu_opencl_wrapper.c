//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <CL/cl.h>									// (in directory specified to compiler)		needed by OpenCL types and functions

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "main.h"								// (in main directory)						needed to recognized input parameters

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "opencl.h"				// (in directory)							needed by device functions
#include "avilib.h"					// (in directory)							needed by avi functions
#include "avimod.h"					// (in directory)							needed by avi functions

//======================================================================================================================================================150
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "kernel_gpu_opencl_wrapper.h"			// (in directory)

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu_opencl_wrapper(	params_common common,
							int* endoRow,
							int* endoCol,
							int* tEndoRowLoc,
							int* tEndoColLoc,
							int* epiRow,
							int* epiCol,
							int* tEpiRowLoc,
							int* tEpiColLoc,
							avi_t* frames)
{

	//======================================================================================================================================================150
	//	CPU VARIABLES
	//======================================================================================================================================================150

	// common variables
	int i;

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	COMMON VARIABLES
	//====================================================================================================100

	// common variables
	int error;

	//====================================================================================================100
	//	GET PLATFORMS (Intel, AMD, NVIDIA, based on provided library), SELECT ONE
	//====================================================================================================100

	// Get number of available platforms
	cl_uint num_platforms;
	error = clGetPlatformIDs(	0, 
								NULL, 
								&num_platforms);		// # of platforms
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	printf("# of platforms %d\n", num_platforms);

	// Get list of available platforms
	cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
	error = clGetPlatformIDs(	num_platforms, 
								platforms, 
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Get names of platforms and print them
	cl_char pbuf[100];
	int plat_count;
	cl_platform_id platform;
	for(plat_count = 0; plat_count < num_platforms; plat_count++){

		platform = platforms[plat_count];

		error = clGetPlatformInfo(	platform, 
							CL_PLATFORM_VENDOR, 
							sizeof(pbuf), 
							pbuf, 
							NULL);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		printf("\tPlatform %d: %s\n", plat_count, pbuf);

	}

	// Select platform
	int plat = 0;
	platform = platforms[plat];
	printf("Selecting platform %d\n", plat);

	//====================================================================================================100
	//	CREATE CONTEXT FOR THE PLATFORM
	//====================================================================================================100

	// Create context properties for selected platform
	cl_context_properties context_properties[3] = {	CL_CONTEXT_PLATFORM, 
													(cl_context_properties) platform, 
													0};

	// Create context for selected platform being GPU
	cl_context context;
	context = clCreateContextFromType(	context_properties, 
										CL_DEVICE_TYPE_ALL, 
										NULL, 
										NULL, 
										&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	GET DEVICES AVAILABLE FOR THE CONTEXT, SELECT ONE
	//====================================================================================================100

	// Get number of devices (previousely selected for the context)
	size_t devices_size;
	error = clGetContextInfo(	context, 
								CL_CONTEXT_DEVICES, 
								0, 
								NULL, 
								&devices_size);		// number of bytes (devices * sizeof(cl_device_id))
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	int num_devices = devices_size / sizeof(cl_device_id);
	printf("# of devices %d\n", num_devices);

	// Get the list of devices (previousely selected for the context)
	cl_device_id* devices = (cl_device_id*)malloc(num_devices*sizeof(cl_device_id));
	error = clGetContextInfo(	context, 
								CL_CONTEXT_DEVICES, 
								devices_size, 
								devices, 
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Get names of devices and print them
	cl_char dbuf[100];
	int devi_count;
	cl_device_id device;
	for(devi_count = 0; devi_count < num_devices; devi_count++){

		device = devices[devi_count];

		error = clGetDeviceInfo(device, 
								CL_DEVICE_NAME, 
								sizeof(dbuf), 
								dbuf, 
								NULL);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		printf("\tDevice %d: %s\n", devi_count, dbuf);

	}

	// Select device (previousely selected for the context) (if there are multiple devices, choose the first one)
	int devi = 0;
	device = devices[devi];
	printf("Selecting device %d\n", devi);

	//====================================================================================================100
	//	CREATE COMMAND QUEUE FOR THE DEVICE
	//====================================================================================================100

	// Create a command queue
	cl_command_queue command_queue;
	command_queue = clCreateCommandQueue(	context, 
											device, 
											0, 
											&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	CREATE PROGRAM, COMPILE IT
	//====================================================================================================100

	// Load kernel source code from file
	const char *source = load_kernel_source("kernel_gpu_opencl.cl");
	size_t sourceSize = strlen(source);

	// Create the program
	cl_program program = clCreateProgramWithSource(	context, 
													1, 
													&source, 
													&sourceSize, 
													&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);


  char clOptions[150];
//  sprintf(clOptions,"-I../../src");                                                    
  sprintf(clOptions,"-I.");
#ifdef RD_WG_SIZE
  sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE=%d", RD_WG_SIZE);
#endif
#ifdef RD_WG_SIZE_0
  sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE_0=%d", RD_WG_SIZE_0);
#endif
#ifdef RD_WG_SIZE_0_0
  sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE_0_0=%d", RD_WG_SIZE_0_0);
#endif
	// Compile the program
	error = clBuildProgram(	program, 
							1, 
							&device, 
							clOptions, 
							NULL, 
							NULL);
	// Print warnings and errors from compilation
	static cl_char log[65536]; 
	memset(log, 0, sizeof(log));
	clGetProgramBuildInfo(	program, 
							device, 
							CL_PROGRAM_BUILD_LOG, 
							sizeof(log)-1, 
							log, 
							NULL);
	printf("-----OpenCL Compiler Output-----\n");
	if (strstr(log,"warning:") || strstr(log, "error:")) 
		printf("<<<<\n%s\n>>>>\n", log);
	printf("--------------------------------\n");
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Create kernel
	cl_kernel kernel;
	kernel = clCreateKernel(program, 
							"kernel_gpu_opencl", 
							&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	TRIGGERING INITIAL DRIVER OVERHEAD
	//====================================================================================================100

	// cudaThreadSynchronize();		// the above does it

	//======================================================================================================================================================150
	//	GPU MEMORY ALLOCATION
	//======================================================================================================================================================150

	//====================================================================================================100
	//	Common	(COPY IN)
	//====================================================================================================100

	cl_mem d_common;
	d_common = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.common_mem,
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	Frame	(COPY IN)
	//====================================================================================================100

	// common
	cl_mem d_frame;
	d_frame = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.frame_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	Inputs	(COPY IN)
	//====================================================================================================100

	//==================================================50
	//	endo points
	//==================================================50

	// common
	cl_mem d_endoRow;
	d_endoRow = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.endo_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_endoCol;
	d_endoCol = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.endo_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_tEndoRowLoc;
	d_tEndoRowLoc = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.endo_mem * common.no_frames, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_tEndoColLoc;
	d_tEndoColLoc = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.endo_mem * common.no_frames, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	epi points
	//==================================================50

	// common
	cl_mem d_epiRow;
	d_epiRow = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.epi_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_epiCol;
	d_epiCol = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.epi_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_tEpiRowLoc;
	d_tEpiRowLoc = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.epi_mem * common.no_frames, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_tEpiColLoc;
	d_tEpiColLoc = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.epi_mem * common.no_frames, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// 	Array of Templates for All Points	(COPY IN)
	//====================================================================================================100

	//==================================================50
	// template sizes
	//==================================================50

	// common
	printf("tSize is %d, sSize is %d\n", common.tSize, common.sSize);
	common.in_rows = common.tSize + 1 + common.tSize;
	common.in_cols = common.in_rows;
	common.in_elem = common.in_rows * common.in_cols;
	common.in_mem = sizeof(fp) * common.in_elem;

	//==================================================50
	// endo points templates
	//==================================================50

	// common
	cl_mem d_endoT;
	d_endoT = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.in_mem * common.endoPoints, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	// epi points templates
	//==================================================50

	// common
	cl_mem d_epiT;
	d_epiT = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.in_mem * common.epiPoints, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// 	AREA AROUND POINT		FROM	FRAME	(LOCAL)
	//====================================================================================================100

	// common
	common.in2_rows = common.sSize + 1 + common.sSize;
	common.in2_cols = common.in2_rows;
	common.in2_elem = common.in2_rows * common.in2_cols;
	common.in2_mem = sizeof(fp) * common.in2_elem;

	// unique
	cl_mem d_in2;
	d_in2 = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							common.in2_mem * common.allPoints, 
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// 	CONVOLUTION	(LOCAL)
	//====================================================================================================100

	// common
	common.conv_rows = common.in_rows + common.in2_rows - 1;												// number of rows in I
	common.conv_cols = common.in_cols + common.in2_cols - 1;												// number of columns in I
	common.conv_elem = common.conv_rows * common.conv_cols;													// number of elements
	common.conv_mem = sizeof(fp) * common.conv_elem;
	common.ioffset = 0;
	common.joffset = 0;

	// unique
	cl_mem d_conv;
	d_conv = clCreateBuffer(context, 
							CL_MEM_READ_WRITE, 
							common.conv_mem * common.allPoints, 
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// 	CUMULATIVE SUM	(LOCAL)
	//====================================================================================================100

	//==================================================50
	// 	PADDING OF ARRAY, VERTICAL CUMULATIVE SUM
	//==================================================50

	// common
	common.in2_pad_add_rows = common.in_rows;
	common.in2_pad_add_cols = common.in_cols;

	common.in2_pad_cumv_rows = common.in2_rows + 2*common.in2_pad_add_rows;
	common.in2_pad_cumv_cols = common.in2_cols + 2*common.in2_pad_add_cols;
	common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
	common.in2_pad_cumv_mem = sizeof(fp) * common.in2_pad_cumv_elem;

	// unique
	cl_mem d_in2_pad_cumv;
	d_in2_pad_cumv = clCreateBuffer(context, 
									CL_MEM_READ_WRITE, 
									common.in2_pad_cumv_mem * common.allPoints, 
									NULL, 
									&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	// 	SELECTION
	//==================================================50

	// common
	common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;													// (1 to n+1)
	common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
	common.in2_pad_cumv_sel_collow = 1;
	common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
	common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
	common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
	common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
	common.in2_pad_cumv_sel_mem = sizeof(fp) * common.in2_pad_cumv_sel_elem;

	// unique
	cl_mem d_in2_pad_cumv_sel;
	d_in2_pad_cumv_sel = clCreateBuffer(context, 
										CL_MEM_READ_WRITE, 
										common.in2_pad_cumv_sel_mem * common.allPoints, 
										NULL, 
										&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	// 	SELECTION	2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
	//==================================================50

	// common
	common.in2_pad_cumv_sel2_rowlow = 1;
	common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
	common.in2_pad_cumv_sel2_collow = 1;
	common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
	common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
	common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
	common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
	common.in2_sub_cumh_mem = sizeof(fp) * common.in2_sub_cumh_elem;

	// unique
	cl_mem d_in2_sub_cumh;
	d_in2_sub_cumh = clCreateBuffer(context, 
									CL_MEM_READ_WRITE, 
									common.in2_sub_cumh_mem * common.allPoints, 
									NULL, 
									&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	// 	SELECTION
	//==================================================50

	// common
	common.in2_sub_cumh_sel_rowlow = 1;
	common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
	common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
	common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
	common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
	common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
	common.in2_sub_cumh_sel_mem = sizeof(fp) * common.in2_sub_cumh_sel_elem;

	// unique
	cl_mem d_in2_sub_cumh_sel;
	d_in2_sub_cumh_sel = clCreateBuffer(context, 
										CL_MEM_READ_WRITE, 
										common.in2_sub_cumh_sel_mem * common.allPoints, 
										NULL, 
										&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	SELECTION 2, SUBTRACTION
	//==================================================50

	// common
	common.in2_sub_cumh_sel2_rowlow = 1;
	common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel2_collow = 1;
	common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
	common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
	common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
	common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
	common.in2_sub2_mem = sizeof(fp) * common.in2_sub2_elem;

	// unique
	cl_mem d_in2_sub2;
	d_in2_sub2 = clCreateBuffer(context, 
								CL_MEM_READ_WRITE, 
								common.in2_sub2_mem * common.allPoints, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	CUMULATIVE SUM 2	(LOCAL)
	//====================================================================================================100

	//==================================================50
	//	MULTIPLICATION
	//==================================================50

	// common
	common.in2_sqr_rows = common.in2_rows;
	common.in2_sqr_cols = common.in2_cols;
	common.in2_sqr_elem = common.in2_elem;
	common.in2_sqr_mem = common.in2_mem;

	// unique
	cl_mem d_in2_sqr;
	d_in2_sqr = clCreateBuffer(context, 
								CL_MEM_READ_WRITE, 
								common.in2_sqr_mem * common.allPoints, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	SELECTION 2, SUBTRACTION
	//==================================================50

	// common
	common.in2_sqr_sub2_rows = common.in2_sub2_rows;
	common.in2_sqr_sub2_cols = common.in2_sub2_cols;
	common.in2_sqr_sub2_elem = common.in2_sub2_elem;
	common.in2_sqr_sub2_mem = common.in2_sub2_mem;

	// unique
	cl_mem d_in2_sqr_sub2;
	d_in2_sqr_sub2 = clCreateBuffer(context, 
									CL_MEM_READ_WRITE, 
									common.in2_sqr_sub2_mem * common.allPoints, 
									NULL, 
									&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	FINAL	(LOCAL)
	//====================================================================================================100

	// common
	common.in_sqr_rows = common.in_rows;
	common.in_sqr_cols = common.in_cols;
	common.in_sqr_elem = common.in_elem;
	common.in_sqr_mem = common.in_mem;

	// unique
	cl_mem d_in_sqr;
	d_in_sqr = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.in_sqr_mem * common.allPoints, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	TEMPLATE MASK CREATE	(LOCAL)
	//====================================================================================================100

	// common
	common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
	common.tMask_cols = common.tMask_rows;
	common.tMask_elem = common.tMask_rows * common.tMask_cols;
	common.tMask_mem = sizeof(fp) * common.tMask_elem;

	// unique
	cl_mem d_tMask;
	d_tMask = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								common.tMask_mem * common.allPoints, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	POINT MASK INITIALIZE	(LOCAL)
	//====================================================================================================100

	// common
	common.mask_rows = common.maxMove;
	common.mask_cols = common.mask_rows;
	common.mask_elem = common.mask_rows * common.mask_cols;
	common.mask_mem = sizeof(fp) * common.mask_elem;

	//====================================================================================================100
	//	MASK CONVOLUTION	(LOCAL)
	//====================================================================================================100

	// common
	common.mask_conv_rows = common.tMask_rows;												// number of rows in I
	common.mask_conv_cols = common.tMask_cols;												// number of columns in I
	common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;												// number of elements
	common.mask_conv_mem = sizeof(fp) * common.mask_conv_elem;
	common.mask_conv_ioffset = (common.mask_rows-1)/2;
	if((common.mask_rows-1) % 2 > 0.5){
		common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
	}
	common.mask_conv_joffset = (common.mask_cols-1)/2;
	if((common.mask_cols-1) % 2 > 0.5){
		common.mask_conv_joffset = common.mask_conv_joffset + 1;
	}

	// unique
	cl_mem d_mask_conv;
	d_mask_conv = clCreateBuffer(	context, 
									CL_MEM_READ_WRITE, 
									common.mask_conv_mem * common.allPoints, 
									NULL, 
									&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	END
	//====================================================================================================100

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	Inputs
	//====================================================================================================100

	//==================================================50
	// endo points
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue, 
									d_endoRow, 
									1, 
									0, 
									common.endo_mem, 
									endoRow, 
									0, 
									0, 
									0);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	error = clEnqueueWriteBuffer(	command_queue, 
									d_endoCol, 
									1, 
									0, 
									common.endo_mem, 
									endoCol, 
									0, 
									0, 
									0);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	// epi points
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue, 
									d_epiRow, 
									1, 
									0, 
									common.epi_mem, 
									epiRow, 
									0, 
									0, 
									0);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	error = clEnqueueWriteBuffer(	command_queue, 
									d_epiCol, 
									1, 
									0, 
									common.epi_mem, 
									epiCol, 
									0, 
									0, 
									0);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	END
	//==================================================50

	//====================================================================================================100
	//	END
	//====================================================================================================100

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	//====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

	// All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
	size_t local_work_size[1];
	local_work_size[0] = NUMBER_THREADS;
	size_t global_work_size[1];
	global_work_size[0] = common.allPoints * local_work_size[0];

	printf("# of workgroups = %d, # of threads/workgroup = %d (ensure that device can handle)\n", (int)(global_work_size[0]/local_work_size[0]), (int)local_work_size[0]);

	//====================================================================================================100
	//	COPY ARGUMENTS
	//====================================================================================================100

	error = clEnqueueWriteBuffer(	command_queue, 
									d_common, 
									1, 
									0, 
									common.common_mem, 
									&common, 
									0, 
									0, 
									0);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	set kernel arguments
	//====================================================================================================100

	// structures
	error = clSetKernelArg(	kernel, 
							0, 
							sizeof(params_common), 
							(void *) &common);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// common
	error = clSetKernelArg(	kernel, 
							3, 
							sizeof(cl_mem), 
							(void *) &d_endoRow);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							4, 
							sizeof(cl_mem), 
							(void *) &d_endoCol);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							5, 
							sizeof(cl_mem), 
							(void *) &d_tEndoRowLoc);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							6, 
							sizeof(cl_mem), 
							(void *) &d_tEndoColLoc);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							7, 
							sizeof(cl_mem), 
							(void *) &d_epiRow);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							8, 
							sizeof(cl_mem), 
							(void *) &d_epiCol);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							9, 
							sizeof(cl_mem), 
							(void *) &d_tEpiRowLoc);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							10, 
							sizeof(cl_mem), 
							(void *) &d_tEpiColLoc);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// common_unique
	error = clSetKernelArg(	kernel, 
							11, 
							sizeof(cl_mem), 
							(void *) &d_endoT);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							12, 
							sizeof(cl_mem), 
							(void *) &d_epiT);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							13, 
							sizeof(cl_mem), 
							(void *) &d_in2);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							14, 
							sizeof(cl_mem), 
							(void *) &d_conv);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							15, 
							sizeof(cl_mem), 
							(void *) &d_in2_pad_cumv);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							16, 
							sizeof(cl_mem), 
							(void *) &d_in2_pad_cumv_sel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							17, 
							sizeof(cl_mem), 
							(void *) &d_in2_sub_cumh);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							18, 
							sizeof(cl_mem), 
							(void *) &d_in2_sub_cumh_sel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							19, 
							sizeof(cl_mem), 
							(void *) &d_in2_sub2);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							20, 
							sizeof(cl_mem), 
							(void *) &d_in2_sqr);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							21, 
							sizeof(cl_mem), 
							(void *) &d_in2_sqr_sub2);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							22, 
							sizeof(cl_mem), 
							(void *) &d_in_sqr);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							23, 
							sizeof(cl_mem), 
							(void *) &d_tMask);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							24, 
							sizeof(cl_mem), 
							(void *) &d_mask_conv);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// // local
	// // int local_size_one;
	// // local_size_one = common.in_rows;
	// error = clSetKernelArg(	kernel, 
							// 25, 
							// // sizeof(fp) * local_size_one, 	// size 51
							// sizeof(fp) * common.in_elem,
							// NULL);
	// if (error != CL_SUCCESS) 
		// fatal_CL(error, __LINE__);
	// error = clSetKernelArg(	kernel, 
							// 26, 
							// // sizeof(fp) * local_size_one, 	// size 51
							// sizeof(fp) * common.in_cols,
							// NULL);
	// if (error != CL_SUCCESS) 
		// fatal_CL(error, __LINE__);
	// // int local_size_two;
	// // local_size_two = common.in_rows + common.in2_rows - 1;
	// error = clSetKernelArg(	kernel, 
							// 27, 
							// // sizeof(fp) * local_size_two, 	// size 51+81-1=131
							// sizeof(fp) * common.in_sqr_rows,
							// NULL);
	// if (error != CL_SUCCESS) 
		// fatal_CL(error, __LINE__);
	// error = clSetKernelArg(	kernel, 
							// 28, 
							// // sizeof(fp) * local_size_two, 	// size 51+81-1=131
							// sizeof(fp) * common.mask_conv_rows,
							// NULL);
	// if (error != CL_SUCCESS) 
		// fatal_CL(error, __LINE__);
	// // int local_size_three;
	// // local_size_three = common.in_rows * common.in_rows;
	// error = clSetKernelArg(	kernel, 
							// 29, 
							// // sizeof(fp) * local_size_three, 	// size 51*51=2601
							// sizeof(int) * common.mask_conv_rows,
							// NULL);
	// if (error != CL_SUCCESS) 
		// fatal_CL(error, __LINE__);

	// int local_size;
	// local_size = (common.in_elem + common.in_cols + common.in_sqr_rows + common.mask_conv_rows) * 4 + common.mask_conv_rows * 2;
	// printf("size of used local memory/workgroup = %dB (ensure that device can handle)\n", local_size);

	cl_mem d_in_mod_temp;
	d_in_mod_temp = clCreateBuffer(	context, 
									CL_MEM_READ_WRITE, 
									sizeof(fp) * common.in_elem * common.allPoints, 
									NULL, 
									&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	cl_mem in_partial_sum;
	in_partial_sum = clCreateBuffer(context, 
										CL_MEM_READ_WRITE, 
										sizeof(fp) * common.in_cols * common.allPoints, 
										NULL, 
										&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	cl_mem in_sqr_partial_sum;
	in_sqr_partial_sum = clCreateBuffer(context, 
										CL_MEM_READ_WRITE, 
										sizeof(fp) * common.in_sqr_rows * common.allPoints, 
										NULL, 
										&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	cl_mem par_max_val;
	par_max_val = clCreateBuffer(	context, 
									CL_MEM_READ_WRITE, 
									sizeof(fp) * common.mask_conv_rows * common.allPoints, 
									NULL, 
									&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	cl_mem par_max_coo;
	par_max_coo = clCreateBuffer(	context, 
									CL_MEM_READ_WRITE, 
									sizeof(int) * common.mask_conv_rows * common.allPoints, 
									NULL, 
									&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	cl_mem in_final_sum;
	in_final_sum = clCreateBuffer(	context, 
									CL_MEM_READ_WRITE, 
									sizeof(fp) * common.allPoints, 
									NULL, 
									&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	cl_mem in_sqr_final_sum;
	in_sqr_final_sum = clCreateBuffer(	context, 
										CL_MEM_READ_WRITE, 
										sizeof(fp) * common.allPoints, 
										NULL, 
										&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	cl_mem denomT;
	denomT = clCreateBuffer(context, 
							CL_MEM_READ_WRITE, 
							sizeof(fp) * common.allPoints, 
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);



	error = clSetKernelArg(	kernel, 
							25, 
							sizeof(cl_mem),
							(void *) &d_in_mod_temp);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							26, 
							sizeof(cl_mem),
							(void *) &in_partial_sum);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							27, 
							sizeof(cl_mem),
							(void *) &in_sqr_partial_sum);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							28, 
							sizeof(cl_mem),
							(void *) &par_max_val);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							29, 
							sizeof(cl_mem),
							(void *) &par_max_coo);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							30, 
							sizeof(cl_mem),
							(void *) &in_final_sum);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							31, 
							sizeof(cl_mem),
							(void *) &in_sqr_final_sum);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	kernel, 
							32, 
							sizeof(cl_mem),
							(void *) &denomT);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);



	cl_mem d_checksum;
	d_checksum = clCreateBuffer(context, 
								CL_MEM_READ_WRITE, 
								sizeof(fp) * CHECK, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	error = clSetKernelArg(	kernel, 
							33, 
							sizeof(cl_mem),
							(void *) &d_checksum);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	PRINT FRAME PROGRESS START
	//====================================================================================================100

	printf("frame progress: ");
	fflush(NULL);

	//====================================================================================================100
	//	LAUNCH
	//====================================================================================================100

	// variables
	fp* frame;
	int frame_no;

	for(frame_no=0; frame_no<common.frames_processed; frame_no++){

		//==================================================50
		//	get and write current frame to GPU buffer
		//==================================================50

		// Extract a cropped version of the first frame from the video file
		frame = get_frame(	frames,								// pointer to video file
							frame_no,							// number of frame that needs to be returned
							0,									// cropped?
							0,									// scaled?
							1);									// converted

		// copy frame to GPU memory
		error = clEnqueueWriteBuffer(	command_queue, 
										d_frame, 
										1, 
										0, 
										common.frame_mem, 
										frame, 
										0, 
										0, 
										0);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		//==================================================50
		//	kernel arguments that change inside this loop
		//==================================================50

		// common_change
		error = clSetKernelArg(	kernel, 
								1, 
								sizeof(cl_mem), 
								(void *) &d_frame);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clSetKernelArg(	kernel, 
								2, 
								sizeof(int), 
								(void *) &frame_no);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		//==================================================50
		//	launch kernel
		//==================================================50

		printf("global_work_size[0]: %d\n", (int)global_work_size[0]);	
		printf("local_work_size[0]: %d\n", (int)local_work_size[0]);

		error = clEnqueueNDRangeKernel(	command_queue, 
										kernel, 
										1, 
										NULL, 
										global_work_size, 
										local_work_size, 
										0, 
										NULL, 
										NULL);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		//==================================================50
		//	finish iteration
		//==================================================50

		// Synchronization, wait for all operations in the command queue so far to finish
		error = clFinish(command_queue);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		// free frame after each loop iteration, since AVI library allocates memory for every frame fetched
		free(frame);

		//==================================================50
		//	print frame progress
		//==================================================50

		// print frame progress
		printf("%d ", frame_no);
		fflush(NULL);

		//==================================================50
		//	DISPLAY CHECKSUM (TESTING)
		//==================================================50

#ifdef TEST_CHECKSUM
		fp* checksum;
		checksum = (fp*)malloc(sizeof(fp) * CHECK);
		error = clEnqueueReadBuffer(command_queue,
									d_checksum,
									CL_TRUE,
									0,
									sizeof(fp)*CHECK,
									checksum,
									0,
									NULL,
									NULL);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		printf("CHECKSUM:\n");
		for(i=0; i<CHECK; i++){
				printf("%f ", checksum[i]);
		}
		printf("\n\n");
#endif

		//==================================================50
		//	End
		//==================================================50

	}

	//====================================================================================================100
	//	PRINT FRAME PROGRESS END
	//====================================================================================================100

	printf("\n");
	fflush(NULL);

	//======================================================================================================================================================150
	//	OUTPUT
	//======================================================================================================================================================150

	//====================================================================================================100
	// endo points
	//====================================================================================================100

	error = clEnqueueReadBuffer(command_queue,
								d_tEndoRowLoc,
								CL_TRUE,
								0,
								common.endo_mem * common.no_frames,
								tEndoRowLoc,
								0,
								NULL,
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// for testing of the output
#ifdef TEST_OUTPUT
	int j;
	for(i=0; i<common.frames_processed; i++){
		printf("%d: ", i);
		for(j=0; j<common.endoPoints; j++){
			printf("%d ", tEndoRowLoc[j*common.no_frames+i]);
		}
		printf("\n\n");
	}
#endif

	error = clEnqueueReadBuffer(command_queue,
								d_tEndoColLoc,
								CL_TRUE,
								0,
								common.endo_mem * common.no_frames,
								tEndoColLoc,
								0,
								NULL,
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// epi points
	//====================================================================================================100

	error = clEnqueueReadBuffer(command_queue,
								d_tEpiRowLoc,
								CL_TRUE,
								0,
								common.epi_mem * common.no_frames,
								tEpiRowLoc,
								0,
								NULL,
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);;

	error = clEnqueueReadBuffer(command_queue,
								d_tEpiColLoc,
								CL_TRUE,
								0,
								common.epi_mem * common.no_frames,
								tEpiColLoc,
								0,
								NULL,
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//======================================================================================================================================================150
	//	DEALLOCATION
	//======================================================================================================================================================150

	// OpenCL structures
	error = clReleaseKernel(kernel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseProgram(program);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// common_change
	error = clReleaseMemObject(d_frame);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// common
	error = clReleaseMemObject(d_endoRow);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_endoCol);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_tEndoRowLoc);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_tEndoColLoc);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_epiRow);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_epiCol);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_tEpiRowLoc);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_tEpiColLoc);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// common_unique
	error = clReleaseMemObject(d_endoT);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_epiT);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_in2);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_conv);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_in2_pad_cumv);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_in2_pad_cumv_sel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_in2_sub_cumh);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_in2_sub_cumh_sel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_in2_sub2);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_in2_sqr);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_in2_sqr_sub2);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_in_sqr);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_tMask);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_mask_conv);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// OpenCL structures
	error = clFlush(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseCommandQueue(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseContext(context);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//======================================================================================================================================================150
	//	End
	//======================================================================================================================================================150

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200
