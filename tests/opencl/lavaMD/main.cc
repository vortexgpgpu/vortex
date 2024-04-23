//#ifdef __cplusplus
//extern "C" {
//#endif

//========================================================================================================================================================================================================200
//======================================================================================================================================================150
//====================================================================================================100
//==================================================50

//========================================================================================================================================================================================================200
//	INFORMATION
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	UPDATE
//======================================================================================================================================================150

//	2009.12 Lukasz G. Szafaryn
//		-- entire code written

//======================================================================================================================================================150
//	DESCRIPTION
//======================================================================================================================================================150

// Description

//======================================================================================================================================================150
//	USE
//======================================================================================================================================================150

// How to run

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>					// (in path known to compiler)			needed by printf
#include <stdlib.h>					// (in path known to compiler)			needed by malloc
#include <stdbool.h>				// (in path known to compiler)			needed by true/false
#include <string.h>
#include <time.h>

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "timer.h"			// (in path specified here)
#include "num.h"				// (in path specified here)

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./main.h"						// (in the current directory)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "kernel_gpu_opencl_wrapper.h"	// (in library path specified here)

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

int platform_id_inuse = 0;            // platform id in use (default: 0)
int device_id_inuse = 0;              // device id in use (default : 0)

int 
main(	int argc, 
		char *argv [])
{

	//======================================================================================================================================================150
	//	CPU/MCPU VARIABLES
	//======================================================================================================================================================150

	// counters
	int i, j, k, l, m, n;

	// system memory
	par_str par_cpu;
	dim_str dim_cpu;
	box_str* box_cpu;
	FOUR_VECTOR* rv_cpu;
	fp* qv_cpu;
	FOUR_VECTOR* fv_cpu;
	int nh;


	printf("WG size of kernel = %d \n", NUMBER_THREADS);

	//======================================================================================================================================================150
	//	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================150

	// assing default values
	dim_cpu.arch_arg = 0;
	dim_cpu.cores_arg = 1;
	dim_cpu.boxes1d_arg = 1;

	// go through arguments
	if (argc >= 3) {
		for(dim_cpu.cur_arg=1; dim_cpu.cur_arg<argc; dim_cpu.cur_arg++){
			// check if -boxes1d
			if(strcmp(argv[dim_cpu.cur_arg], "-boxes1d")==0){
				// check if value provided
				if(argc>=dim_cpu.cur_arg+1){
					// check if value is a number
					if(isInteger(argv[dim_cpu.cur_arg+1])==1){
						dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg+1]);
						if(dim_cpu.boxes1d_arg<0){
							printf("ERROR: Wrong value to -boxes1d argument, cannot be <=0\n");
							return 0;
						}
						dim_cpu.cur_arg = dim_cpu.cur_arg+1;
					}
					// value is not a number
					else{
						printf("ERROR: Value to -boxes1d argument in not a number\n");
						return 0;
					}
				}
				// value not provided
				else{
					printf("ERROR: Missing value to -boxes1d argument\n");
					return 0;
				}
			}
            else if(strcmp(argv[dim_cpu.cur_arg], "-p")==0){
				if(argc>=dim_cpu.cur_arg+1){
				    platform_id_inuse = atoi(argv[dim_cpu.cur_arg+1]);
					dim_cpu.cur_arg = dim_cpu.cur_arg+1;
				}
			}
            else if(strcmp(argv[dim_cpu.cur_arg], "-d")==0){
				if(argc>=dim_cpu.cur_arg+1){
				    device_id_inuse = atoi(argv[dim_cpu.cur_arg+1]);
					dim_cpu.cur_arg = dim_cpu.cur_arg+1;
				}
			}
			// unknown
			else{
				printf("ERROR: Unknown argument\n");
				return 0;
			}
		}
		// Print configuration
		printf("Configuration used: arch = %d, cores = %d, boxes1d = %d\n", dim_cpu.arch_arg, dim_cpu.cores_arg, dim_cpu.boxes1d_arg);
	}
	else{
		printf("Provide boxes1d argument, example: -boxes1d 16 [-p platform_id] [-d device_id]");
		return 0;
	}

	//======================================================================================================================================================150
	//	INPUTS
	//======================================================================================================================================================150

	par_cpu.alpha = 0.5;

	//======================================================================================================================================================150
	//	DIMENSIONS
	//======================================================================================================================================================150

	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg; // 8*8*8=512

	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;							//512*100=51,200
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

	//======================================================================================================================================================150
	//	SYSTEM MEMORY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	BOX
	//====================================================================================================100

	// allocate boxes
	box_cpu = (box_str*)malloc(dim_cpu.box_mem);

	// initialize number of home boxes
	nh = 0;

	// home boxes in z direction
	for(i=0; i<dim_cpu.boxes1d_arg; i++){
		// home boxes in y direction
		for(j=0; j<dim_cpu.boxes1d_arg; j++){
			// home boxes in x direction
			for(k=0; k<dim_cpu.boxes1d_arg; k++){

				// current home box
				box_cpu[nh].x = k;
				box_cpu[nh].y = j;
				box_cpu[nh].z = i;
				box_cpu[nh].number = nh;
				box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

				// initialize number of neighbor boxes
				box_cpu[nh].nn = 0;

				// neighbor boxes in z direction
				for(l=-1; l<2; l++){
					// neighbor boxes in y direction
					for(m=-1; m<2; m++){
						// neighbor boxes in x direction
						for(n=-1; n<2; n++){

							// check if (this neighbor exists) and (it is not the same as home box)
							if(		(((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)	&&
									(l==0 && m==0 && n==0)==false	){

								// current neighbor box
								box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
								box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
								box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
								box_cpu[nh].nei[box_cpu[nh].nn].number =	(box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) + 
																			(box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) + 
																			 box_cpu[nh].nei[box_cpu[nh].nn].x;
								box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

								// increment neighbor box
								box_cpu[nh].nn = box_cpu[nh].nn + 1;

							}

						} // neighbor boxes in x direction
					} // neighbor boxes in y direction
				} // neighbor boxes in z direction

				// increment home box
				nh = nh + 1;

			} // home boxes in x direction
		} // home boxes in y direction
	} // home boxes in z direction

	//====================================================================================================100
	//	PARAMETERS, DISTANCE, CHARGE AND FORCE
	//====================================================================================================100

	// random generator seed set to random value - time in this case
	srand(time(NULL));

	// input (distances)
	rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		rv_cpu[i].v = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		// rv_cpu[i].v = 0.1;			// get a number in the range 0.1 - 1.0
		rv_cpu[i].x = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		// rv_cpu[i].x = 0.2;			// get a number in the range 0.1 - 1.0
		rv_cpu[i].y = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		// rv_cpu[i].y = 0.3;			// get a number in the range 0.1 - 1.0
		rv_cpu[i].z = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		// rv_cpu[i].z = 0.4;			// get a number in the range 0.1 - 1.0
	}

	// input (charge)
	qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		qv_cpu[i] = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		// qv_cpu[i] = 0.5;			// get a number in the range 0.1 - 1.0
	}

	// output (forces)
	fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		fv_cpu[i].v = 0;								// set to 0, because kernels keeps adding to initial value
		fv_cpu[i].x = 0;								// set to 0, because kernels keeps adding to initial value
		fv_cpu[i].y = 0;								// set to 0, because kernels keeps adding to initial value
		fv_cpu[i].z = 0;								// set to 0, because kernels keeps adding to initial value
	}

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU_OPENCL
	//====================================================================================================100

	kernel_gpu_opencl_wrapper(	par_cpu,
								dim_cpu,
								box_cpu,
								rv_cpu,
								qv_cpu,
								fv_cpu);

	//======================================================================================================================================================150
	//	SYSTEM MEMORY DEALLOCATION
	//======================================================================================================================================================150

	// dump results
#ifdef OUTPUT
        FILE *fptr;
	fptr = fopen("result.txt", "w");	
	for(i=0; i<dim_cpu.space_elem; i=i+1){
        	fprintf(fptr, "%f, %f, %f, %f\n", fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
	}
	fclose(fptr);
#endif       	


	free(rv_cpu);
	free(qv_cpu);
	free(fv_cpu);
	free(box_cpu);

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	// printf("Time spent in different stages of the application:\n");

	//======================================================================================================================================================150
	//	RETURN
	//======================================================================================================================================================150

	return 0.0;																					// always returns 0.0

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

//#ifdef __cplusplus
//}
//#endif
