// Tier-A bilinear image read. A small CL_RGBA/UNORM_INT8 power-of-two image is
// upsampled through a linear, normalized, clamp-to-edge sampler — the output
// grid is finer than the input, so most taps land between texels and exercise
// real interpolation. On a VX_CFG_EXT_TEX_ENABLE device read_imagef issues a
// single hardware vx_tex4 (bilinear) sample where software would do four texel
// loads plus a weighted blend; on a device without TEX it samples in software.
// Results are written to a global byte buffer for host comparison.
__kernel void image_ff_linear(read_only image2d_t src,
                              sampler_t sampler,
                              __global uchar* out,
                              int out_w,
                              int out_h) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= out_w || y >= out_h)
    return;
  float u = (x + 0.5f) / (float)out_w;
  float v = (y + 0.5f) / (float)out_h;
  float4 t = read_imagef(src, sampler, (float2)(u, v));
  int idx = (y * out_w + x) * 4;
  out[idx + 0] = (uchar)(t.x * 255.0f + 0.5f);
  out[idx + 1] = (uchar)(t.y * 255.0f + 0.5f);
  out[idx + 2] = (uchar)(t.z * 255.0f + 0.5f);
  out[idx + 3] = (uchar)(t.w * 255.0f + 0.5f);
}
