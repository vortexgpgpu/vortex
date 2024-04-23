#ifdef DOUBLE_PRECISION
	#pragma OPENCL EXTENSION cl_khr_fp64: enable
	#define FP_TYPE double
	#define FP_CONST(num) num
#else
	#define FP_TYPE float
	#define FP_CONST(num) num##f
#endif

// Constants used in the MGVF computation
#define PI FP_CONST(3.14159)
#define ONE_OVER_PI (FP_CONST(1.0) / PI)
#define MU FP_CONST(0.5)
#define LAMBDA (FP_CONST(8.0) * MU + FP_CONST(1.0))

// The number of threads per thread block
#define LOCAL_WORK_SIZE 256
#define NEXT_LOWEST_POWER_OF_TWO 256


// Regularized version of the Heaviside step function:
// He(x) = (atan(x) / pi) + 0.5
FP_TYPE heaviside(FP_TYPE x) {
	return (atan(x) * ONE_OVER_PI) + FP_CONST(0.5);

	// A simpler, faster approximation of the Heaviside function
	/* float out = 0.0;
	if (x > -0.0001) out = 0.5;
	if (x >  0.0001) out = 1.0;
	return out; */
}


// Kernel to compute the Motion Gradient Vector Field (MGVF) matrix for multiple cells
__kernel void IMGVF_kernel(__global FP_TYPE *IMGVF_array, __global FP_TYPE *I_array, __constant int *I_offsets, int __constant *m_array,
						   __constant int *n_array, FP_TYPE vx, FP_TYPE vy, FP_TYPE e, int max_iterations, FP_TYPE cutoff) {
	
	// Shared copy of the matrix being computed
	__local FP_TYPE IMGVF[41 * 81];
	
	// Shared buffer used for two purposes:
	// 1) To temporarily store newly computed matrix values so that only
	//    values from the previous iteration are used in the computation.
	// 2) To store partial sums during the tree reduction which is performed
	//    at the end of each iteration to determine if the computation has converged.
	__local FP_TYPE buffer[LOCAL_WORK_SIZE];
	
	// Figure out which cell this thread block is working on
	int cell_num = get_group_id(0);
	
	// Get pointers to current cell's input image and inital matrix
	int I_offset = I_offsets[cell_num];
	__global FP_TYPE *IMGVF_global = &(IMGVF_array[I_offset]);
	__global FP_TYPE *I = &(I_array[I_offset]);
	
	// Get current cell's matrix dimensions
	int m = m_array[cell_num];
	int n = n_array[cell_num];
	
	// Compute the number of virtual thread blocks
	int IMGVF_Size = m * n;
	int tb_count = (IMGVF_Size + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE;
	
	// Load the initial IMGVF matrix into shared memory
	int thread_id = get_local_id(0);
	int thread_block, i, j;
	for (thread_block = 0; thread_block < tb_count; thread_block++) {
		int offset = thread_block * LOCAL_WORK_SIZE + thread_id;
		if (offset < IMGVF_Size)
			IMGVF[offset] = IMGVF_global[offset];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Set the converged flag to false
	__local int cell_converged;
	if (thread_id == 0) cell_converged = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Constants used to iterate through virtual thread blocks
	const float one_nth = 1.0f / (float) n;
	const int tid_mod = thread_id % n;
	const int tbsize_mod = LOCAL_WORK_SIZE % n;
	
	// Constant used in the computation of Heaviside values
	FP_TYPE one_over_e = FP_CONST(1.0) / e;
	
	// Iteratively compute the IMGVF matrix until the computation has
	//  converged or we have reached the maximum number of iterations
	int iterations = 0;
	while ((! cell_converged) && (iterations < max_iterations)) {
	
		// The total change to this thread's matrix elements in the current iteration
		FP_TYPE total_diff = FP_CONST(0.0);
		
		int old_i = 0, old_j = 0;
		j = tid_mod - tbsize_mod;
		
		// Iterate over virtual thread blocks
		for (thread_block = 0; thread_block < tb_count; thread_block++) {
			// Store the index of this thread's previous matrix element
			//  (used in the buffering scheme below)
			old_i = i;
			old_j = j;
			
			// Determine the index of this thread's current matrix element 
			int offset = thread_block * LOCAL_WORK_SIZE;
			i = (thread_id + offset) * one_nth;
			j += tbsize_mod;
			if (j >= n) j -= n;
			
			FP_TYPE new_val, old_val;
			
			// Make sure the thread has not gone off the end of the matrix
			if (i < m) {
				// Compute neighboring matrix element indices
				int rowU = (i == 0) ? 0 : i - 1;
				int rowD = (i == m - 1) ? m - 1 : i + 1;
				int colL = (j == 0) ? 0 : j - 1;
				int colR = (j == n - 1) ? n - 1 : j + 1;
				
				// Compute the difference between the matrix element and its eight neighbors
				old_val = IMGVF[(i * n) + j];
				FP_TYPE U  = IMGVF[(rowU * n) + j   ] - old_val;
				FP_TYPE D  = IMGVF[(rowD * n) + j   ] - old_val;
				FP_TYPE L  = IMGVF[(i    * n) + colL] - old_val;
				FP_TYPE R  = IMGVF[(i    * n) + colR] - old_val;
				FP_TYPE UR = IMGVF[(rowU * n) + colR] - old_val;
				FP_TYPE DR = IMGVF[(rowD * n) + colR] - old_val;
				FP_TYPE UL = IMGVF[(rowU * n) + colL] - old_val;
				FP_TYPE DL = IMGVF[(rowD * n) + colL] - old_val;
				
				// Compute the regularized heaviside value for these differences
				FP_TYPE UHe  = heaviside((U  *       -vy)  * one_over_e);
				FP_TYPE DHe  = heaviside((D  *        vy)  * one_over_e);
				FP_TYPE LHe  = heaviside((L  *  -vx     )  * one_over_e);
				FP_TYPE RHe  = heaviside((R  *   vx     )  * one_over_e);
				FP_TYPE URHe = heaviside((UR * ( vx - vy)) * one_over_e);
				FP_TYPE DRHe = heaviside((DR * ( vx + vy)) * one_over_e);
				FP_TYPE ULHe = heaviside((UL * (-vx - vy)) * one_over_e);
				FP_TYPE DLHe = heaviside((DL * (-vx + vy)) * one_over_e);
				
				// Update the IMGVF value in two steps:
				// 1) Compute IMGVF += (mu / lambda)(UHe .*U  + DHe .*D  + LHe .*L  + RHe .*R +
				//                                   URHe.*UR + DRHe.*DR + ULHe.*UL + DLHe.*DL);
				new_val = old_val + (MU / LAMBDA) * (UHe  * U  + DHe  * D  + LHe  * L  + RHe  * R +
													 URHe * UR + DRHe * DR + ULHe * UL + DLHe * DL);
				// 2) Compute IMGVF -= (1 / lambda)(I .* (IMGVF - I))
				FP_TYPE vI = I[(i * n) + j];
				new_val -= ((1.0 / LAMBDA) * vI * (new_val - vI));

				// Keep track of the total change of this thread's matrix elements
				total_diff += fabs(new_val - old_val);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			// Save the previous virtual thread block's value (if it exists)
			if (thread_block > 0) {
				offset = (thread_block - 1) * LOCAL_WORK_SIZE;
				if (old_i < m) IMGVF[(old_i * n) + old_j] = buffer[thread_id];
			}
			if (thread_block < tb_count - 1) {
				// Write the new value to the buffer
				buffer[thread_id] = new_val;
			} else {
				// We've reached the final virtual thread block,
				//  so write directly to the matrix
				if (i < m) IMGVF[(i * n) + j] = new_val;
			}
			
			// We need to synchronize between virtual thread blocks to prevent
			//  threads from writing the values from the buffer to the actual
			//  IMGVF matrix too early
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		
		// We need to compute the overall sum of the change at each matrix element
		//  by performing a tree reduction across the whole threadblock
		buffer[thread_id] = total_diff;
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// Account for thread block sizes that are not a power of 2
		if (thread_id >= NEXT_LOWEST_POWER_OF_TWO) {
			buffer[thread_id - NEXT_LOWEST_POWER_OF_TWO] += buffer[thread_id];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// Perform the tree reduction
		int th;
		for (th = NEXT_LOWEST_POWER_OF_TWO / 2; th > 0; th /= 2) {
			if (thread_id < th) {
				buffer[thread_id] += buffer[thread_id + th];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		
		// Figure out if we have converged
		if(thread_id == 0) {
			FP_TYPE mean = buffer[thread_id] / (FP_TYPE) (m * n);
			if (mean < cutoff) {
				// We have converged, so set the appropriate flag
				cell_converged = 1;
			}
		}
		
		// We need to synchronize to ensure that all threads
		//  read the correct value of the convergence flag
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// Keep track of the number of iterations we have performed
		iterations++;
	}
	
	// Save the final IMGVF matrix to global memory
	for (thread_block = 0; thread_block < tb_count; thread_block++) {
		int offset = thread_block * LOCAL_WORK_SIZE + thread_id;
		if (offset < IMGVF_Size)
			IMGVF_global[offset] = IMGVF[offset];
	}
}
