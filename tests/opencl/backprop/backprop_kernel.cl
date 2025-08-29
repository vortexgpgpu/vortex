#define THREADS 256
#define WIDTH 16  
#define HEIGHT 16 
#define ETA 0.3f       
#define MOMENTUM 0.3f  

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_
#define WM(i, j)   weight_matrix[(j) + (i) * WIDTH]

__kernel void 
bpnn_layerforward_ocl(__global float *input_cuda,
	                  __global float *output_hidden_cuda,
					  __global float *input_hidden_cuda,
					  __global float *hidden_partial_sum,
					  __local float *input_node,
					  __local float *weight_matrix,
					  int in,
					  int hid) 
{

   int by = get_group_id(1);
   int tx = get_local_id(0);
   int ty = get_local_id(1);

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

   int index_in = HEIGHT * by + ty + 1;
   
	if ( tx == 0 )
		input_node[ty] = input_cuda[index_in] ;
		barrier(CLK_LOCAL_MEM_FENCE);

		weight_matrix[ty * WIDTH + tx] =  input_hidden_cuda[index];
		barrier(CLK_LOCAL_MEM_FENCE);
   
		weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] * input_node[ty];
		barrier(CLK_LOCAL_MEM_FENCE);
   
		for ( int i = 1 ; i <= HEIGHT ; i=i*2){
	//for ( int i = 1 ; i <= 4 ; i++){
      int power_two = i; 
		//int power_two = 2 << (i - 1);

	    if( ty % power_two == 0 )
		  weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] + weight_matrix[(ty + power_two/2)* WIDTH + tx];
		  
		barrier(CLK_LOCAL_MEM_FENCE);

    }
   
    input_hidden_cuda[index] =  weight_matrix[ty * WIDTH + tx];
   
	barrier(CLK_LOCAL_MEM_FENCE);

    if ( tx == 0 ) {
	  hidden_partial_sum[by * hid + ty] = weight_matrix[tx* WIDTH + ty];
    }

}


__kernel void  bpnn_adjust_weights_ocl( __global float * delta,   
										 int hid,         
										__global float * ly,      
										 int in,          
										__global float * w,       
										__global float * oldw)  									
{
   
   int by = get_group_id(1);
   int tx = get_local_id(0);
   int ty = get_local_id(1);
	
   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;

   w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

   barrier(CLK_LOCAL_MEM_FENCE);

   if (ty == 0 && by ==0){
	w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
	oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   }

}
#endif 
