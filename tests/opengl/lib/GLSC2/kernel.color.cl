__kernel void rgba4 (
  __global unsigned short *colorbuffer,
  const unsigned int width,
  //
  __global const uint3 *gl_position,
  __global bool *gl_discard,
  __global float4 *gl_fragcolor
) {
  int gid = get_global_id(0);
  
  if (gl_discard[gid]) return;

  float4 color = gl_fragcolor[gid];

  unsigned short value = 0;
  value |= (unsigned short) (color[0] * 0xF);
  value |= (unsigned short) (color[0] * 0xF) << 4;
  value |= (unsigned short) (color[0] * 0xF) << 8;
  value |= (unsigned short) (color[0] * 0xF) << 12;

  colorbuffer[gl_position[gid][1]*width + gl_position[gid][0]] = value;
}