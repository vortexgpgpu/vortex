__kernel void vecadd (__global const float *A,
	                    __global const float *B,
	                    __global float *C)
{
  int gid = get_global_id(0);
  C[gid] = A[gid] + B[gid];
}