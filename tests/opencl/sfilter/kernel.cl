// m0 m1 m2
// m3 m4 m5
// m6 m7 m8
__kernel void sfilter(__global float *src, __global float *dst, long ldc,
  float m0, float m1, float m2, float m3, float m4, float m5, float m6, float m7, float m8)
{
  long x = get_global_id(0);
  long y = get_global_id(1);

  int addr = x + y * ldc;
  
  float i0 = src[addr-1-1*ldc]*m0;
  float i1 = src[addr+0-1*ldc]*m1;
  float i2 = src[addr+1-1*ldc]*m2;
  float i3 = src[addr-1+0*ldc]*m3;
  float i4 = src[addr+0+0*ldc]*m4;
  float i5 = src[addr+1+0*ldc]*m5;
  float i6 = src[addr-1+1*ldc]*m6;
  float i7 = src[addr+0+1*ldc]*m7;
  float i8 = src[addr+1+1*ldc]*m8;
  
  dst[addr] = i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8;
}
