//#ifdef __cplusplus
//extern "C" {
//#endif

//========================================================================================================================================================================================================200
//	INCLUDE/DEFINE (had to bring from ./../main.h here because feature of including headers in clBuildProgram does not work for some reason)
//========================================================================================================================================================================================================200

// #include <main.h>						// (in the directory SOMEHOW known to the OpenCL compiler function)

#define fp float

#define NUMBER_PAR_PER_BOX 100							// keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

#define NUMBER_THREADS 128								// this should be roughly equal to NUMBER_PAR_PER_BOX for best performance

#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))	// STABLE

//===============================================================================================================================================================================================================200
//	STRUCTURES (had to bring from ./../main.h here because feature of including headers in clBuildProgram does not work for some reason)
//===============================================================================================================================================================================================================200

typedef struct
{
	fp x, y, z;

} THREE_VECTOR;

typedef struct
{
	fp v, x, y, z;

} FOUR_VECTOR;

typedef struct nei_str
{

	// neighbor box
	int x, y, z;
	int number;
	long offset;

} nei_str;

typedef struct box_str
{

	// home box
	int x, y, z;
	int number;
	long offset;

	// neighbor boxes
	int nn;
	nei_str nei[26];

} box_str;

typedef struct par_str
{

	fp alpha;

} par_str;

typedef struct dim_str
{

	// input arguments
	int cur_arg;
	int arch_arg;
	int cores_arg;
	int boxes1d_arg;

	// system memory
	long number_boxes;
	long box_mem;
	long space_elem;
	long space_mem;
	long space_mem2;

} dim_str;

//========================================================================================================================================================================================================200
//	kernel_gpu_opencl KERNEL
//========================================================================================================================================================================================================200

__kernel void kernel_gpu_opencl(	par_str d_par_gpu,
					dim_str d_dim_gpu,
					__global box_str *d_box_gpu,
					__global FOUR_VECTOR *d_rv_gpu,
					__global fp *d_qv_gpu,
					__global FOUR_VECTOR *d_fv_gpu)
{

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	THREAD PARAMETERS
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

	int bx = get_group_id(0);															// get current horizontal block index (0-n)
	int tx = get_local_id(0);															// get current horizontal thread index (0-n)
	int wtx = tx;

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	ALLOCATE LOCAL VARIABLES
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

    // (enable the lines below only if wanting to use shared memory)
    __local FOUR_VECTOR rA_shared[100];
    __local FOUR_VECTOR rB_shared[100];
    __local fp qB_shared[100];

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	DO FOR THE NUMBER OF BOXES
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

	if(bx<d_dim_gpu.number_boxes){

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	Extract input parameters
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		// parameters
		fp a2 = 2*d_par_gpu.alpha*d_par_gpu.alpha;

		// home box
		int first_i;

		// nei box
		int pointer;
		int k = 0;
		int first_j;
		int j = 0;

		// common
		fp r2;
		fp u2;
		fp vij;
		fp fs;
		fp fxij;
		fp fyij;
		fp fzij;
		THREE_VECTOR d;

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	Home box
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Setup parameters
		//----------------------------------------------------------------------------------------------------------------------------------140

		// home box - box parameters
		first_i = d_box_gpu[bx].offset;

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Copy to shared memory
		//----------------------------------------------------------------------------------------------------------------------------------140

		// (enable the section below only if wanting to use shared memory)
		// home box - shared memory
		while(wtx<NUMBER_PAR_PER_BOX){
			rA_shared[wtx] = d_rv_gpu[first_i+wtx];
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// (enable the section below only if wanting to use shared memory)
		// synchronize threads  - not needed, but just to be safe for now
		barrier(CLK_LOCAL_MEM_FENCE);

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		// loop over nei boxes of home box
		for (k=0; k<(1+d_box_gpu[bx].nn); k++){

			//----------------------------------------50
			//	nei box - get pointer to the right box
			//----------------------------------------50

			if(k==0){
				pointer = bx;													// set first box to be processed to home box
			}
			else{
				pointer = d_box_gpu[bx].nei[k-1].number;							// remaining boxes are nei boxes
			}

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Setup parameters
			//----------------------------------------------------------------------------------------------------------------------------------140

			// nei box - box parameters
			first_j = d_box_gpu[pointer].offset;

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Setup parameters
			//----------------------------------------------------------------------------------------------------------------------------------140

			// (enable the section below only if wanting to use shared memory)
			// nei box - shared memory
			while(wtx<NUMBER_PAR_PER_BOX){
				rB_shared[wtx] = d_rv_gpu[first_j+wtx];
				qB_shared[wtx] = d_qv_gpu[first_j+wtx];
				wtx = wtx + NUMBER_THREADS;
			}
			wtx = tx;

			// (enable the section below only if wanting to use shared memory)
			// synchronize threads because in next section each thread accesses data brought in by different threads here
			barrier(CLK_LOCAL_MEM_FENCE);

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation
			//----------------------------------------------------------------------------------------------------------------------------------140

			// loop for the number of particles in the home box
			while(wtx<NUMBER_PAR_PER_BOX){

				// loop for the number of particles in the current nei box
				for (j=0; j<NUMBER_PAR_PER_BOX; j++){

					// (disable the section below only if wanting to use shared memory)
					// r2 = d_rv_gpu[first_i+wtx].v + d_rv_gpu[first_j+j].v - DOT(d_rv_gpu[first_i+wtx],d_rv_gpu[first_j+j]); 
					// u2 = a2*r2;
					// vij= exp(-u2);
					// fs = 2*vij;
					// d.x = d_rv_gpu[first_i+wtx].x  - d_rv_gpu[first_j+j].x;
					// fxij=fs*d.x;
					// d.y = d_rv_gpu[first_i+wtx].y  - d_rv_gpu[first_j+j].y;
					// fyij=fs*d.y;
					// d.z = d_rv_gpu[first_i+wtx].z  - d_rv_gpu[first_j+j].z;
					// fzij=fs*d.z;
					// d_fv_gpu[first_i+wtx].v +=  d_qv_gpu[first_j+j]*vij;
					// d_fv_gpu[first_i+wtx].x +=  d_qv_gpu[first_j+j]*fxij;
					// d_fv_gpu[first_i+wtx].y +=  d_qv_gpu[first_j+j]*fyij;
					// d_fv_gpu[first_i+wtx].z +=  d_qv_gpu[first_j+j]*fzij;

					// (enable the section below only if wanting to use shared memory)
					r2 = rA_shared[wtx].v + rB_shared[j].v - DOT(rA_shared[wtx],rB_shared[j]); 
					u2 = a2*r2;
					vij= exp(-u2);
					fs = 2*vij;
					d.x = rA_shared[wtx].x  - rB_shared[j].x;
					fxij=fs*d.x;
					d.y = rA_shared[wtx].y  - rB_shared[j].y;
					fyij=fs*d.y;
					d.z = rA_shared[wtx].z  - rB_shared[j].z;
					fzij=fs*d.z;
					d_fv_gpu[first_i+wtx].v +=  qB_shared[j]*vij;
					d_fv_gpu[first_i+wtx].x +=  qB_shared[j]*fxij;
					d_fv_gpu[first_i+wtx].y +=  qB_shared[j]*fyij;
					d_fv_gpu[first_i+wtx].z +=  qB_shared[j]*fzij;

				}

				// increment work thread index
				wtx = wtx + NUMBER_THREADS;

			}

			// reset work index
			wtx = tx;

			// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
			barrier(CLK_LOCAL_MEM_FENCE);

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation END
			//----------------------------------------------------------------------------------------------------------------------------------140

		}

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop END
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

	}

}

//========================================================================================================================================================================================================200
//	END kernel_gpu_opencl KERNEL
//========================================================================================================================================================================================================200

//#ifdef __cplusplus
//}
//#endif
