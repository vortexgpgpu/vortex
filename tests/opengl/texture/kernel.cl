typedef struct sampler2d {
  int width, height;
  __global const float4* mem;
} sampler2d;


__kernel void gl_main_vs (
  // my imp
  __global const float* positions,
  __global const float2* texCoord,
  // implementation values
  __global float4 *gl_Positions,
  __global float4 *gl_Primitives
) {
  int gid = get_global_id(0);
  // in out vars
  float x = positions[gid*3];
  float y = positions[gid*3+1];
  float z = positions[gid*3+2];

  // vertex operations
  gl_Positions[gid] = (float4) (x, y, z, 1.0f);
  gl_Primitives[gid*2] = (float4) (texCoord[gid].x,texCoord[gid].y, 1.0f, 1.0f);
}


float4 texture2d(struct sampler2d texture, float4 texCoord) {
  int w = texture.width * texCoord.x;
  int h = texture.height * texCoord.y;
  return texture.mem[h*texture.width + w];
}

__kernel void gl_main_fs (
  // user values
  struct sampler2d texture, 
  // implementation values 
  __global float4 *gl_FragCoord, // position of the fragment in the window space, z is depth value
  __global const float4 *gl_Rasterization,
  __global bool *gl_Discard, // if discarded
  __global float4 *gl_FragColor // out color of the fragment || It is deprecated in OpenGL 3.0 
)
{
  int gid = get_global_id(0);
  // in out vars
  float4 texCoord = gl_Rasterization[gid*2];
  // fragment operations
  gl_FragColor[gid] = texture2d(texture, texCoord);
}
