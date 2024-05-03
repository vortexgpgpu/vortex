
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
  gl_Primitives[gid*2] = (float4) (texCoord[gid].x, texCoord[gid].y, 1.0f, 1.0f);
}


float4 texture2d(int2 size, __global const unsigned char* texture, float2 texCoord) {
  int w = (int) (size.x * texCoord.x) % size.x;
  int h = size.y - ((int) (size.y * texCoord.y) % size.y) - 1;
  __global const unsigned char* color = texture + (h*size.x + w)*4;
  
  return (float4) ((float)*color / 255, (float)*(color+1) / 255, (float)*(color+2) / 255, (float)*(color+3) / 255);
}

__kernel void gl_main_fs (
  // user values
  #ifndef IMAGE_SUPPORT
  const int2 size,
  __global const unsigned char *image,
  #else
  sampler_t sampler,
  read_only image2d_t image,
  #endif
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
  #ifndef IMAGE_SUPPORT
  gl_FragColor[gid] = texture2d(size, image, (float2) (texCoord.x, texCoord.y));
  #else
  gl_FragColor[gid] = read_imagef(image, sampler, (float2) (texCoord.x, texCoord.y));
  #endif
}
