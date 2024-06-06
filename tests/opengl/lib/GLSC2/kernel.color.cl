__kernel void gl_rgba4 (
  __global unsigned short *gl_Color,
  __global bool *gl_Discard,
  __global float4 *gl_FragColor
) {
  int gid = get_global_id(0);
  
  if (gl_Discard[gid]) return;

  float4 color = gl_FragColor[gid];

  unsigned short value = 0; // HW optimization ??
  value |= (unsigned short) (color.x * 0xFu) << 0;
  value |= (unsigned short) (color.y * 0xFu) << 4;
  value |= (unsigned short) (color.z * 0xFu) << 8;
  value |= (unsigned short) (color.w * 0xFu) << 12;

  gl_Color[gid] = value;
}

__kernel void gl_rgba8 (
  __global unsigned int *gl_Color,
  __global bool *gl_Discard,
  __global float4 *gl_FragColor
) {
  int gid = get_global_id(0);
  
  if (gl_Discard[gid]) return;

  float4 color = gl_FragColor[gid];

  unsigned int value = 0; // HW optimization ??
  value |= (unsigned int) (color.x * 0xFFu) << 0;
  value |= (unsigned int) (color.y * 0xFFu) << 8;
  value |= (unsigned int) (color.z * 0xFFu) << 16;
  value |= (unsigned int) (color.w * 0xFFu) << 24;

  gl_Color[gid] = value;
}