__kernel void saxpy(__global float *src, __global float *dst, float factor)
{
  long i = get_global_id(0);
  dst[i] += src[i] * factor;
}
