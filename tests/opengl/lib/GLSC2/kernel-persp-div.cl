
__kernel void perspective_division(__global const float *cc,
	                    __global float *ndc)
{
  int gid = get_global_id(0);
  float w = cc[4*gid+3];

  float* write_ndc = &ndc[3*gid];
  
  *ndc++=(*cc)/w;
  *ndc++=(*cc)/w;
  *ndc=(*cc)/w;
}
