#ifdef __cplusplus
extern "C" {
#endif

//===============================================================================================================================================================================================================200
//	INCLUDE/DEFINE
//===============================================================================================================================================================================================================200

#include <stdio.h>					// (in path known to compiler)		needed by printf

#include <CL/cl.h>					// (in path specified to compiler)	needed by OpenCL types

//===============================================================================================================================================================================================================200
//	INCLUDE/DEFINE
//===============================================================================================================================================================================================================200

#include "opencl.h"					// (in directory)	function headers

//===============================================================================================================================================================================================================200
//	LOAD KERNEL SOURCE CODE FUNCTION
//===============================================================================================================================================================================================================200

char *
load_kernel_source(const char *filename)
{
	// Open the source file
	FILE *file = fopen(filename, "r");
	if (file == NULL){
		fatal("Error opening kernel source file\n");
	}

	// Determine the size of the file
	if (fseek(file, 0, SEEK_END)){
		fatal("Error reading kernel source file\n");
	}
	size_t size = ftell(file);

	// Allocate space for the source code (plus one for null-terminator)
	char *source = (char *) malloc(size + 1);

	// Read the source code into the string
	fseek(file, 0, SEEK_SET);
	// printf("Number of elements: %lu\nSize = %lu", fread(source, 1, size, file), size);
	// exit(1);
	if (fread(source, 1, size, file) != size){
		fatal("Error reading kernel source file\n");
	}

	// Null-terminate the string
	source[size] = '\0';

	// Return the pointer to the string
	return source;
}

//===============================================================================================================================================================================================================200
//	PRINT ERROR FUNCTION
//===============================================================================================================================================================================================================200

void 
fatal(const char *s)
{
	fprintf(stderr, "Error: %s\n", s);
	exit(1);
}

//===============================================================================================================================================================================================================200
//	PRINT OPENCL ERROR FUNCTION
//===============================================================================================================================================================================================================200

void 
fatal_CL(cl_int error, int line_no) {

	printf("Error at line %d: ", line_no);

	switch(error) {

		case CL_SUCCESS: 									printf("CL_SUCCESS\n"); break;
		case CL_DEVICE_NOT_FOUND: 							printf("CL_DEVICE_NOT_FOUND\n"); break;
		case CL_DEVICE_NOT_AVAILABLE: 						printf("CL_DEVICE_NOT_AVAILABLE\n"); break;
		case CL_COMPILER_NOT_AVAILABLE: 					printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: 				printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
		case CL_OUT_OF_RESOURCES: 							printf("CL_OUT_OF_RESOURCES\n"); break;
		case CL_OUT_OF_HOST_MEMORY: 						printf("CL_OUT_OF_HOST_MEMORY\n"); break;
		case CL_PROFILING_INFO_NOT_AVAILABLE: 				printf("CL_PROFILING_INFO_NOT_AVAILABLE\n"); break;
		case CL_MEM_COPY_OVERLAP: 							printf("CL_MEM_COPY_OVERLAP\n"); break;
		case CL_IMAGE_FORMAT_MISMATCH: 						printf("CL_IMAGE_FORMAT_MISMATCH\n"); break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: 				printf("CL_IMAGE_FORMAT_NOT_SUPPORTED\n"); break;
		case CL_BUILD_PROGRAM_FAILURE: 						printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
		case CL_MAP_FAILURE: 								printf("CL_MAP_FAILURE\n"); break;
		case CL_INVALID_VALUE: 								printf("CL_INVALID_VALUE\n"); break;
		case CL_INVALID_DEVICE_TYPE: 						printf("CL_INVALID_DEVICE_TYPE\n"); break;
		case CL_INVALID_PLATFORM: 							printf("CL_INVALID_PLATFORM\n"); break;
		case CL_INVALID_DEVICE: 							printf("CL_INVALID_DEVICE\n"); break;
		case CL_INVALID_CONTEXT: 							printf("CL_INVALID_CONTEXT\n"); break;
		case CL_INVALID_QUEUE_PROPERTIES: 					printf("CL_INVALID_QUEUE_PROPERTIES\n"); break;
		case CL_INVALID_COMMAND_QUEUE: 						printf("CL_INVALID_COMMAND_QUEUE\n"); break;
		case CL_INVALID_HOST_PTR: 							printf("CL_INVALID_HOST_PTR\n"); break;
		case CL_INVALID_MEM_OBJECT: 						printf("CL_INVALID_MEM_OBJECT\n"); break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: 			printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR\n"); break;
		case CL_INVALID_IMAGE_SIZE: 						printf("CL_INVALID_IMAGE_SIZE\n"); break;
		case CL_INVALID_SAMPLER: 							printf("CL_INVALID_SAMPLER\n"); break;
		case CL_INVALID_BINARY: 							printf("CL_INVALID_BINARY\n"); break;
		case CL_INVALID_BUILD_OPTIONS: 						printf("CL_INVALID_BUILD_OPTIONS\n"); break;
		case CL_INVALID_PROGRAM: 							printf("CL_INVALID_PROGRAM\n"); break;
		case CL_INVALID_PROGRAM_EXECUTABLE: 				printf("CL_INVALID_PROGRAM_EXECUTABLE\n"); break;
		case CL_INVALID_KERNEL_NAME: 						printf("CL_INVALID_KERNEL_NAME\n"); break;
		case CL_INVALID_KERNEL_DEFINITION: 					printf("CL_INVALID_KERNEL_DEFINITION\n"); break;
		case CL_INVALID_KERNEL: 							printf("CL_INVALID_KERNEL\n"); break;
		case CL_INVALID_ARG_INDEX: 							printf("CL_INVALID_ARG_INDEX\n"); break;
		case CL_INVALID_ARG_VALUE: 							printf("CL_INVALID_ARG_VALUE\n"); break;
		case CL_INVALID_ARG_SIZE: 							printf("CL_INVALID_ARG_SIZE\n"); break;
		case CL_INVALID_KERNEL_ARGS: 						printf("CL_INVALID_KERNEL_ARGS\n"); break;
		case CL_INVALID_WORK_DIMENSION: 					printf("CL_INVALID_WORK_DIMENSION\n"); break;
		case CL_INVALID_WORK_GROUP_SIZE: 					printf("CL_INVALID_WORK_GROUP_SIZE\n"); break;
		case CL_INVALID_WORK_ITEM_SIZE: 					printf("CL_INVALID_WORK_ITEM_SIZE\n"); break;
		case CL_INVALID_GLOBAL_OFFSET: 						printf("CL_INVALID_GLOBAL_OFFSET\n"); break;
		case CL_INVALID_EVENT_WAIT_LIST: 					printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
		case CL_INVALID_EVENT: 								printf("CL_INVALID_EVENT\n"); break;
		case CL_INVALID_OPERATION: 							printf("CL_INVALID_OPERATION\n"); break;
		case CL_INVALID_GL_OBJECT: 							printf("CL_INVALID_GL_OBJECT\n"); break;
		case CL_INVALID_BUFFER_SIZE: 						printf("CL_INVALID_BUFFER_SIZE\n"); break;
		case CL_INVALID_MIP_LEVEL: 							printf("CL_INVALID_MIP_LEVEL\n"); break;
		case CL_INVALID_GLOBAL_WORK_SIZE: 					printf("CL_INVALID_GLOBAL_WORK_SIZE\n"); break;

		#ifdef CL_VERSION_1_1
		case CL_MISALIGNED_SUB_BUFFER_OFFSET: 				printf("CL_MISALIGNED_SUB_BUFFER_OFFSET\n"); break;
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n"); break;
		/* case CL_INVALID_PROPERTY: 							printf("CL_INVALID_PROPERTY\n"); break; */
		#endif

		default:											printf("Invalid OpenCL error code\n");

	}

	exit(error);

}

//===============================================================================================================================================================================================================200
//	END
//===============================================================================================================================================================================================================200

#ifdef __cplusplus
}
#endif
