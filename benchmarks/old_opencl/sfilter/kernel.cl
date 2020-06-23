// m0 m1 m2
// m3 m4 m5
// m6 m7 m8
__kernel void sfilter(__global float *src, __global float *dst, long ldc,
  float m0, float m1, float m2, float m3, float m4, float m5, float m6, float m7, float m8)
{
  long x = get_global_id(0);
  long y = get_global_id(1);
  
  float i0 = src[(x-1)+(y-1)*ldc]*m0;
  float i1 = src[(x)  +(y-1)*ldc]*m1;
  float i2 = src[(x+1)+(y-1)*ldc]*m2;
  float i3 = src[(x-1)+(y)  *ldc]*m3;
  float i4 = src[(x)  + y  * ldc]*m4;
  float i5 = src[(x+1)+(y)  *ldc]*m5;
  float i6 = src[(x-1)+(y+1)*ldc]*m6;
  float i7 = src[(x)  +(y+1)*ldc]*m7;
  float i8 = src[(x+1)+(y+1)*ldc]*m8;

  dst[x+y*ldc] = i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8;
}
