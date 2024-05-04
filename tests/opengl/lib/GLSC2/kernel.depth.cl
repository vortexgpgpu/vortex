__kernel void gl_never (
  __global unsigned short *gl_Depth,
  __global bool *gl_Discard,
  __global const float4 *gl_FragCoord
) {
  int gid = get_global_id(0);
  gl_Discard[gid] = true;
}

__kernel void gl_always (
  __global unsigned short *gl_Depth,
  __global bool *gl_Discard,
  __global const float4 *gl_FragCoord
) {
  int gid = get_global_id(0);
  if (gl_Discard[gid]) return;

  gl_Depth[gid] = gl_FragCoord[gid].z * 0xFFFFu;
}

__kernel void gl_less (
  __global unsigned short *gl_Depth,
  __global bool *gl_Discard,
  __global const float4 *gl_FragCoord
) {
  int gid = get_global_id(0);
  if (gl_Discard[gid]) return;

  unsigned short z = gl_FragCoord[gid].z * 0xFFFFu;

  if (z < gl_Depth[gid]) gl_Depth[gid] = z;
  else gl_Discard[gid] = true;
}

__kernel void gl_lequal (
  __global unsigned short *gl_Depth,
  __global bool *gl_Discard,
  __global const float4 *gl_FragCoord
) {
  int gid = get_global_id(0);
  if (gl_Discard[gid]) return;

  unsigned short z = gl_FragCoord[gid].z * 0xFFFFu;

  if (z <= gl_Depth[gid]) gl_Depth[gid] = z;
  else gl_Discard[gid] = true;
}

__kernel void gl_equal (
  __global unsigned short *gl_Depth,
  __global bool *gl_Discard,
  __global const float4 *gl_FragCoord
) {
  int gid = get_global_id(0);
  if (gl_Discard[gid]) return;

  unsigned short z = gl_FragCoord[gid].z * 0xFFFFu;

  if (z == gl_Depth[gid]) gl_Depth[gid] = z;
  else gl_Discard[gid] = true;
}

__kernel void gl_greater (
  __global unsigned short *gl_Depth,
  __global bool *gl_Discard,
  __global const float4 *gl_FragCoord
) {
  int gid = get_global_id(0);
  if (gl_Discard[gid]) return;

  unsigned short z = gl_FragCoord[gid].z * 0xFFFFu;

  if (z > gl_Depth[gid]) gl_Depth[gid] = z;
  else gl_Discard[gid] = true;
}

__kernel void gl_gequal (
  __global unsigned short *gl_Depth,
  __global bool *gl_Discard,
  __global const float4 *gl_FragCoord
) {
  int gid = get_global_id(0);
  if (gl_Discard[gid]) return;

  unsigned short z = gl_FragCoord[gid].z * 0xFFFFu;

  if (z >= gl_Depth[gid]) gl_Depth[gid] = z;
  else gl_Discard[gid] = true;
}

__kernel void gl_notequal (
  __global unsigned short *gl_Depth,
  __global bool *gl_Discard,
  __global const float4 *gl_FragCoord
) {
  int gid = get_global_id(0);
  if (gl_Discard[gid]) return;

  unsigned short z = gl_FragCoord[gid].z * 0xFFFFu;

  if (z != gl_Depth[gid]) gl_Depth[gid] = z;
  else gl_Discard[gid] = true;
}