float4 texture2d(int2 size, __global const unsigned char* texture, float2 texCoord) {
  int w = (int) (size.x * texCoord.x) % size.x;
  int h = size.y - ((int) (size.y * texCoord.y) % size.y) - 1;
  __global const unsigned char* color = texture + (h*size.x + w)*4;
  
  return (float4) ((float)*color / 255, (float)*(color+1) / 255, (float)*(color+2) / 255, (float)*(color+3) / 255);
}

__kernel void gl_main_fs (
  // user values
  const int2 size,
  __global const unsigned char *image,
  // implementation values 
  __global float4 *gl_FragCoord, // position of the fragment in the window space, z is depth value
  __global const float4 *gl_Rasterization,
  __global bool *gl_Discard, // if discarded
  __global float4 *gl_FragColor // out color of the fragment || It is deprecated in OpenGL 3.0 
)
{
  int gid = get_global_id(0);
  // in out vars
  float4 texCoord = gl_Rasterization[gid];
  // fragment operations
  gl_FragColor[gid] = texture2d(size, image, (float2) (texCoord.x, texCoord.y));
}
