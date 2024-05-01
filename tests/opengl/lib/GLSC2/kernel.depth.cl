__kernel void gl_less (
  __global unsigned short *gl_Depth,
  __global bool *gl_Discard,
  __global const float4 *gl_FragCoord
) {

  int gid = get_global_id(0);
  
  if (gl_Discard[gid]) return;

  unsigned short z = 0xFFFFu * gl_FragCoord[gid].z;

  if (z < gl_Depth[gid]) {
    gl_Depth[gid] = z;
  } else {
    gl_Discard[gid] = true;
  }

}