__kernel void psort (__global const float *in, __global float *out)
{
  int gid = get_global_id(0);
  int n = get_global_size(0);

  float ref = in[gid];
  
  int pos = 0;
  for (int i = 0; i < n; ++i) {
    float cur = in[i];
    pos += (cur < ref) || (cur == ref && i < gid);
  }
  out[pos] = ref;
}