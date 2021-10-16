__kernel void psorti (__global const int *in, __global int *out)
{
  int gid = get_global_id(0);
  int n = get_global_size(0);

  int ref = in[gid];
  
  int pos = 0;
  for (int i = 0; i < n; ++i) {
    int cur = in[i];
    pos += (cur < ref) || ((cur == ref) && (i < gid));
  }
  out[pos] = ref;
}

__kernel void psortf (__global const float *in, __global float *out)
{
  int gid = get_global_id(0);
  int n = get_global_size(0);

  float ref = in[gid];
  
  int pos = 0;
  for (int i = 0; i < n; ++i) {
    float cur = in[i];
    pos += (cur < ref) || ((cur == ref) && (i < gid));
    /*int cl = (cur < ref);
    int ce = (cur == ref);
    int ls = (i < gid);
    int x = ce && ls;
    int y = cl || x;
    pos += y;*/
  }
  out[pos] = ref;
}