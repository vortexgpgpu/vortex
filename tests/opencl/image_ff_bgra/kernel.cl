// Tier-A (fixed-function TEX) image read. `src` is a 2D CL_RGBA/UNORM_INT8
// power-of-two image sampled through a nearest, unnormalized, clamp-to-edge
// sampler passed as a kernel argument — exactly the envelope the Vortex FF TEX
// unit accelerates. On a VX_CFG_EXT_TEX_ENABLE device the PoCL driver binds the
// image to a TEX stage and read_imagef issues a hardware vx_tex4 sample; on a
// device without TEX it samples in software. Either way integer coordinates
// address the exact texel, so a UNORM_INT8 round-trip is bit-exact. The result
// is written to a plain global byte buffer so the host can compare raw bytes.
__kernel void image_ff_bgra(read_only image2d_t src,
                       sampler_t sampler,
                       __global uchar* out) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int w = get_image_width(src);
  int h = get_image_height(src);
  if (x >= w || y >= h)
    return;
  float4 t = read_imagef(src, sampler, (int2)(x, y));
  int idx = (y * w + x) * 4;
  out[idx + 0] = (uchar)(t.x * 255.0f + 0.5f);
  out[idx + 1] = (uchar)(t.y * 255.0f + 0.5f);
  out[idx + 2] = (uchar)(t.z * 255.0f + 0.5f);
  out[idx + 3] = (uchar)(t.w * 255.0f + 0.5f);
}
