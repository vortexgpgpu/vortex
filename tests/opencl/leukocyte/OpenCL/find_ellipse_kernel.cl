// The number of sample points in each ellipse (stencil)
#define NPOINTS 150
// The maximum radius of a sample ellipse
#define MAX_RAD 20
// The total number of sample ellipses
#define NCIRCLES 7
// The size of the structuring element used in dilation
#define STREL_SIZE (12 * 2 + 1)



// Kernel to find the maximal GICOV value at each pixel of a
//  video frame, based on the input x- and y-gradient matrices
#ifdef USE_IMAGE
__kernel void GICOV_kernel(int grad_m, image2d_t grad_x, image2d_t grad_y, __constant float *c_sin_angle,
                           __constant float *c_cos_angle, __constant int *c_tX, __constant int *c_tY, __global float *gicov) {
#else
__kernel void GICOV_kernel(int grad_m, __global float *grad_x, __global float *grad_y, __constant float *c_sin_angle,
                           __constant float *c_cos_angle, __constant int *c_tX, __constant int *c_tY, __global float *gicov, int width, int height) {
#endif
	
	int i, j, k, n, x, y;
	int gid = get_global_id(0);
	if(gid>=width*height)
	  return;
	
	// Determine this thread's pixel
	i = gid/width + MAX_RAD + 2;
	j = gid%width + MAX_RAD + 2;

	// Initialize the maximal GICOV score to 0
	float max_GICOV = 0.f;
	
	#ifdef USE_IMAGE
	// Define the sampler for accessing the images
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	#endif

	// Iterate across each stencil
	for (k = 0; k < NCIRCLES; k++) {
		// Variables used to compute the mean and variance
		//  of the gradients along the current stencil
		float sum = 0.f, M2 = 0.f, mean = 0.f;		
		
		// Iterate across each sample point in the current stencil
		for (n = 0; n < NPOINTS; n++) {
			// Determine the x- and y-coordinates of the current sample point
			y = j + c_tY[(k * NPOINTS) + n];
			x = i + c_tX[(k * NPOINTS) + n];
			
			// Compute the combined gradient value at the current sample point
			#ifdef USE_IMAGE
			int2 addr = {y, x};
			float p = read_imagef(grad_x, sampler, addr).x * c_cos_angle[n] + 
			          read_imagef(grad_y, sampler, addr).x * c_sin_angle[n];
			#else
			int addr = x * grad_m + y;
			float p = grad_x[addr] * c_cos_angle[n] + grad_y[addr] * c_sin_angle[n];
			#endif
			
			// Update the running total
			sum += p;
			
			// Partially compute the variance
			float delta = p - mean;
			mean = mean + (delta / (float) (n + 1));
			M2 = M2 + (delta * (p - mean));
		}
		
		// Finish computing the mean
		mean = sum / ((float) NPOINTS);
		
		// Finish computing the variance
		float var = M2 / ((float) (NPOINTS - 1));
		
		// Keep track of the maximal GICOV value seen so far
		if (((mean * mean) / var) > max_GICOV) max_GICOV = (mean * mean) / var;
	}
	
	// Store the maximal GICOV value
	gicov[(i * grad_m) + j] = max_GICOV;
}


// Kernel to compute the dilation of the GICOV matrix produced by the GICOV kernel
// Each element (i, j) of the output matrix is set equal to the maximal value in
//  the neighborhood surrounding element (i, j) in the input matrix
// Here the neighborhood is defined by the structuring element (c_strel)
#ifdef USE_IMAGE
__kernel void dilate_kernel(int img_m, int img_n, int strel_m, int strel_n, __constant float *c_strel,
                            image2d_t img, __global float *dilated) {
#else
__kernel void dilate_kernel(int img_m, int img_n, int strel_m, int strel_n, __constant float *c_strel,
                            __global float *img, __global float *dilated) {
#endif
	
	// Find the center of the structuring element
	int el_center_i = strel_m / 2;
	int el_center_j = strel_n / 2;

	// Determine this thread's location in the matrix
	int thread_id = get_global_id(0); //(blockIdx.x * blockDim.x) + threadIdx.x;
	int i = thread_id % img_m;
	int j = thread_id / img_m;

	if(j > img_n) return;

	// Initialize the maximum GICOV score seen so far to zero
	float max = 0.0f;
	
	#ifdef USE_IMAGE
	// Define the sampler for accessing the image
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	#endif

	// Iterate across the structuring element in one dimension
	int el_i, el_j, x, y;
	// Lingjie Zhang modificated at 11/06/2015

    if (j < img_n){
        for (el_i = 0; el_i < strel_m; el_i++) {
	    	y = i - el_center_i + el_i;
	    	// Make sure we have not gone off the edge of the matrix
	    	if ( (y >= 0) && (y < img_m) ) {
	    		// Iterate across the structuring element in the other dimension
	    		for (el_j = 0; el_j < strel_n; el_j++) {
	    			x = j - el_center_j + el_j;
	    			// Make sure we have not gone off the edge of the matrix
	    			//  and that the current structuring element value is not zero
	    			if ( (x >= 0) &&
	    				 (x < img_n) &&
	    				 (c_strel[(el_i * strel_n) + el_j] != 0) ) {
	    					// Determine if this is the maximal value seen so far
	    					#ifdef USE_IMAGE
	    					int2 addr = {y, x};
	    					float temp = read_imagef(img, sampler, addr).x;
	    					#else
	    					int addr = (x * img_m) + y;
	    					float temp = img[addr];
	    					#endif
	    					if (temp > max) max = temp;
	    			}
	    		}
	    	}
	    }
	    
	    // Store the maximum value found
	    dilated[(i * img_n) + j] = max;
    }
    // end of Lingjie Zhang's modification
}
