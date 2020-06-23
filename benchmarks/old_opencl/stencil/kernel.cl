/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#define Index3D(_nx,_ny,_i,_j,_k) ((_i)+_nx*((_j)+_ny*(_k)))

__kernel void naive_kernel(float c0,float c1,__global float* A0,__global float *Anext,int nx,int ny,int nz)
{
    	int i = get_global_id(0)+1;
    	int j = get_global_id(1)+1;
    	int k = get_global_id(2)+1;

if(i<nx-1)
{
		Anext[Index3D (nx, ny, i, j, k)] = c1 *
		( A0[Index3D (nx, ny, i, j, k + 1)] +
		  A0[Index3D (nx, ny, i, j, k - 1)] +
		  A0[Index3D (nx, ny, i, j + 1, k)] +
		  A0[Index3D (nx, ny, i, j - 1, k)] +
	 	  A0[Index3D (nx, ny, i + 1, j, k)] +
		  A0[Index3D (nx, ny, i - 1, j, k)] )
		- A0[Index3D (nx, ny, i, j, k)] * c0;
}
}
