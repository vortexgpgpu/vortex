__kernel void gl_less (
  __global unsigned short *depthbuffer,
  const unsigned int width,
  //
  __global const uint3 *gl_position,
  __global unsigned short *gl_depth,
  __global bool *gl_discard
) {
  int gid = get_global_id(0);
  
  if (gl_discard[gid]) return;

  unsigned short value = depthbuffer[gl_position[gid][1]*width + gl_position[gid][0]]; 

  if (value < gl_depth[gid]) {
    gl_depth[gid] = value;
  } else {
    gl_discard[gid] = true;
  }

}