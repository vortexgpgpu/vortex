// Bilinear-filtered sample at horizontal midpoints: read_imagef with a
// CLK_FILTER_LINEAR sampler at (x+1.0, y+0.5) returns the 50/50 average of
// texels (x,y) and (x+1,y) under clamp-to-edge. Result written to a float buffer.
__kernel void image_linear(read_only image2d_t src, __global float* out,
                           sampler_t sampler, int W) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  float4 c = read_imagef(src, sampler, (float2)(x + 1.0f, y + 0.5f));
  int i = (y * W + x) * 4;
  out[i + 0] = c.x; out[i + 1] = c.y; out[i + 2] = c.z; out[i + 3] = c.w;
}
