// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//======================================================================================================================================================150
//====================================================================================================100
//==================================================50

//========================================================================================================================================================================================================200
//	UPDATE
//========================================================================================================================================================================================================200

// Lukasz G. Szafaryn 24 JAN 09

//========================================================================================================================================================================================================200
//	DESCRIPTION
//========================================================================================================================================================================================================200

// Myocyte application models cardiac myocyte (heart muscle cell) and simulates its behavior according to the work by Saucerman and Bers [8]. The model integrates 
// cardiac myocyte electrical activity with the calcineurin pathway, which is a key aspect of the development of heart failure. The model spans large number of temporal 
// scales to reflect how changes in heart rate as observed during exercise or stress contribute to calcineurin pathway activation, which ultimately leads to the expression 
// of numerous genes that remodel the heart�s structure. It can be used to identify potential therapeutic targets that may be useful for the treatment of heart failure. 
// Biochemical reactions, ion transport and electrical activity in the cell are modeled with 91 ordinary differential equations (ODEs) that are determined by more than 200 
// experimentally validated parameters. The model is simulated by solving this group of ODEs for a specified time interval. The process of ODE solving is based on the 
// causal relationship between values of ODEs at different time steps, thus it is mostly sequential. At every dynamically determined time step, the solver evaluates the 
// model consisting of a set of 91 ODEs and 480 supporting equations to determine behavior of the system at that particular time instance. If evaluation results are not 
// within the expected tolerance at a given time step (usually as a result of incorrect determination of the time step), another calculation attempt is made at a modified 
// (usually reduced) time step. Since the ODEs are stiff (exhibit fast rate of change within short time intervals), they need to be simulated at small time scales with an 
// adaptive step size solver. 

//	1) The original version of the current solver code was obtained from: Mathematics Source Library (http://mymathlib.webtrellis.net/index.html). The solver has been 
//      somewhat modified to tailor it to our needs. However, it can be reverted back to original form or modified to suit other simulations.
// 2) This solver and particular solving algorithm used with it (embedded_fehlberg_7_8) were adapted to work with a set of equations, not just one like in original version.
//	3) In order for solver to provide deterministic number of steps (needed for particular amount of memore previousely allocated for results), every next step is 
//      incremented by 1 time unit (h_init).
//	4) Function assumes that simulation starts at some point of time (whatever time the initial values are provided for) and runs for the number of miliseconds (xmax) 
//      specified by the uses as a parameter on command line.
// 5) The appropriate amount of memory is previousely allocated for that range (y).
//	6) This setup in 3) - 5) allows solver to adjust the step ony from current time instance to current time instance + 0.9. The next time instance is current time instance + 1;
//	7) The original solver cannot handle cases when equations return NAN and INF values due to discontinuities and /0. That is why equations provided by user need to 
//      make sure that no NAN and INF are returned.
// 8) Application reads initial data and parameters from text files: y.txt and params.txt respectively that need to be located in the same folder as source files. 
//     For simplicity and testing purposes only, when multiple number of simulation instances is specified, application still reads initial data from the same input files. That 
//     can be modified in this source code.

//========================================================================================================================================================================================================200
//	IMPLEMENTATION-SPECIFIC DESCRIPTION (CUDA)
//========================================================================================================================================================================================================200

// This is the CUDA version of Myocyte code.

// The original single-threaded code was written in MATLAB and used MATLAB ode45 ODE solver. In the process of accelerating this code, we arrived with the 
// intermediate versions that used single-threaded Sundials CVODE solver which evaluated model parallelized with CUDA at each time step. In order to convert entire 
// solver to CUDA code (to remove some of the operational overheads such as kernel launches and data transfer in CUDA) we used a simpler solver, from Mathematics 
// Source Library, and tailored it to our needs. The parallelism in the cardiac myocyte model is on a very fine-grained level, close to that of ILP, therefore it is very hard 
// to exploit as DLP or TLB in CUDA code. We were able to divide the model into 4 individual groups that run in parallel. However, even that is not enough work to 
// compensate for some of the CUDA thread launch and data transfer overheads which resulted in performance worse than that of single-threaded C code. Speedup in 
// this code could be achieved only if a customizable accelerator such as FPGA was used for evaluation of the model itself. We also approached the application from 
// another angle and allowed it to run several concurrent simulations, thus turning it into an embarrassingly parallel problem. This version of the code is also useful for 
// scientists who want to run the same simulation with different sets of input parameters. Speedup achieved with CUDA code is variable on the other hand. It depends on 
// the number of concurrent simulations and it saturates around 300 simulations with the speedup of about 10x.

// Speedup numbers reported in the description of this application were obtained on the machine with: Intel Quad Core CPU, 4GB of RAM, Nvidia GTX280 GPU.  

// 1) When running with parallelization inside each simulation instance (value of 3rd command line parameter equal to 0), performance is bad because:
// a) underutilization of GPU (only 1 out of 32 threads in each block)
// b) significant CPU-GPU memory copy overhead
// c) kernel launch overhead (kernel needs to be finished every time model is evaluated as it is the only way to synchronize threads in different blocks)
// 2) When running with parallelization across simulation instances, code gets continues speedup with the increasing number of simulation insances which saturates
//     around 240 instances on GTX280 (roughly corresponding to the number of multiprocessorsXprocessors in GTX280), with the speedup of around 10x compared
//     to serial C version of code. Limited performance is explained mainly by:
// a) significant CPU-GPU memory copy overhead
// b) increasingly uncoalesced memory accesses with the increasing number of workloads
// c) lack of cache compared to CPU, or no use of shared memory to compensate
// d) frequency of GPU shader being lower than that of CPU core
// 3) GPU version has an issue with memory allocation that has not been resolved yet. For certain simulation ranges memory allocation fails, or pointers incorrectly overlap 
//     causeing value trashing.

// The following are the command parameters to the application:
// 1) Simulation time interval which is the number of miliseconds to simulate. Needs to be integer > 0
// Example:
// ./a.out -time 100

//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "./common.h"										// (in directory)

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "file.h"								// (in directory)
#include "timer.h"								// (in directory)
#include "num.h"									// (in directory)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "kernel_gpu_opencl_wrapper.h"				// (in directory)

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>											// (in path known to compiler) needed by printf
#include <stdlib.h>											// (in path known to compiler) meeded by malloc, free

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "./main.h"											// (in directory)

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

int 
main(	int argc, 
		char *argv []){

	//======================================================================================================================================================150
	// 	INPUT ARGUMENTS
	//======================================================================================================================================================150

	// assing default values
	int cur_arg;
	int xmax;
	int workload = 1;

	// go through arguments
	if(argc==3){
		for(cur_arg=1; cur_arg<argc; cur_arg++){
			// check if -time
			if(strcmp(argv[cur_arg], "-time")==0){
				// check if value provided
				if(argc>=cur_arg+1){
					// check if value is a number
					if(isInteger(argv[cur_arg+1])==1){
						xmax = atoi(argv[cur_arg+1]);
						if(xmax<0){
							printf("ERROR: Wrong value to -time argument, cannot be <=0\n");
							return 0;
						}
						cur_arg = cur_arg+1;
					}
					// value is not a number
					else{
						printf("ERROR: Value to -time argument in not a number\n");
						return 0;
					}
				}
				// value not provided
				else{
					printf("ERROR: Missing value to -time argument\n");
					return 0;
				}
			}
			// unknown
			else{
				printf("ERROR: Unknown argument\n");
				return 0;
			}
		}
		// Print configuration
		//printf("Configuration used: arch = %d, cores = %d, time = %d\n", arch_arg, cores_arg, xmax);
	}
	else{
		printf("Provide time argument, example: -time 100");
		return 0;
	}

	//======================================================================================================================================================150
	//	EXECUTION IF THERE IS 1 WORKLOAD, PARALLELIZE INSIDE 1 WORKLOAD
	//======================================================================================================================================================150

	if(workload == 1){

		//====================================================================================================100
		//	VARIABLES
		//====================================================================================================100

		int i,j;

		//====================================================================================================100
		//	MEMORY CHECK
		//====================================================================================================100

		long long memory;
		memory = workload*(xmax+1)*EQUATIONS*4;
		if(memory>1000000000){
			printf("ERROR: trying to allocate more than 1.0GB of memory, decrease workload and span parameters or change memory parameter\n");
			return 0;
		}

		//====================================================================================================100
		//	ALLOCATE ARRAYS
		//====================================================================================================100

		fp*** y;
		y = (fp ***) malloc(workload* sizeof(fp **));
		for(i=0; i<workload; i++){
			y[i] = (fp**)malloc((1+xmax)*sizeof(fp*));
			for(j=0; j<(1+xmax); j++){
				y[i][j]= (fp *) malloc(EQUATIONS* sizeof(fp));
			}
		}

		fp** x;
		x = (fp **) malloc(workload * sizeof(fp *));
		for (i= 0; i<workload; i++){
			x[i]= (fp *)malloc((1+xmax) *sizeof(fp));
		}

		fp** params;
		params = (fp **) malloc(workload * sizeof(fp *));
		for (i= 0; i<workload; i++){
			params[i]= (fp *)malloc(PARAMETERS * sizeof(fp));
		}

		fp* com;
		com = (fp*)malloc(3 * sizeof(fp));

		//====================================================================================================100
		//	INITIAL VALUES
		//====================================================================================================100

		// y
		for(i=0; i<workload; i++){
			read_file( "../../data/myocyte/y.txt",
						y[i][0],
						EQUATIONS,
						1,
						0);
		}

		// params
		for(i=0; i<workload; i++){
			read_file("../../data/myocyte/params.txt",
						params[i],
						PARAMETERS,
						1,
						0);
		}

		//====================================================================================================100
		//	COMPUTATION
		//====================================================================================================100

		kernel_gpu_opencl_wrapper(	xmax,					// span
									workload,				// # of workloads

									y,
									x,
									params,
									com);


	  FILE * pFile;
	  pFile = fopen ("output.txt","w");
	  if (pFile==NULL)
	    {
	  fputs ("fopen example",pFile);
	  return -1;
	}
	  // print results
	  int k;
	  for(i=0; i<workload; i++){
	  fprintf(pFile, "WORKLOAD %d:\n", i);
	  for(j=0; j<(xmax+1); j++){
	  fprintf(pFile, "\tTIME %d:\n", j);
	  for(k=0; k<EQUATIONS; k++){
	  fprintf(pFile, "\t\ty[%d][%d][%d]=%10.7e\n", i, j, k, y[i][j][k]);
	}
	}
	}

	  fclose (pFile);
	  


		//====================================================================================================100
		//	FREE SYSTEM MEMORY
		//====================================================================================================100

		// y values
		for (i= 0; i< workload; i++){
			for (j= 0; j< (1+xmax); j++){
				free(y[i][j]);
			}
			free(y[i]);
		}
		free(y);

		// x values
		for (i= 0; i< workload; i++){
			free(x[i]);
		}
		free(x);

		// parameters
		for (i= 0; i< workload; i++){
			free(params[i]);
		}
		free(params);

		// com
		free(com);

		//====================================================================================================100
		//	END
		//====================================================================================================100

	}

	//======================================================================================================================================================150
	//	RETURN
	//======================================================================================================================================================150

	return 0;

	//======================================================================================================================================================150
	//	END
	//======================================================================================================================================================150

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif
