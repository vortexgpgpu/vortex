// Tier-equivalence probe. Three identical CL_RGBA/UNORM_INT8 images are sampled
// with one sampler and written to three separate buffers. On a device with the
// fixed-function TEX unit the PoCL driver binds the first two images to the two
// TEX stages (hardware vx_tex4, Tier A) and the third overflows the stage budget
// and samples in software (Tier C) — so a==b==c asserts that the hardware path
// matches the software path across whatever filter/wrap the sampler selects, and
// exercises both stages plus the graceful software fallback in one launch.
//
// Coordinates are normalized and deliberately swept across [-0.5, 1.5) so wrap
// modes are exercised on both sides of the image: repeat/mirror fold, and
// clamp-to-edge saturates at BOTH the low edge (negative coords -> texel 0) and
// the high edge. The negative-clamp case is also the regression guard for the
// RTL TEX saturation fix (VX_tex_sat sign-gated overflow).
__kernel void image_hwsw(read_only image2d_t a,
                         read_only image2d_t b,
                         read_only image2d_t c,
                         sampler_t sampler,
                         __global uchar* oa,
                         __global uchar* ob,
                         __global uchar* oc,
                         int ow,
                         int oh) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= ow || y >= oh)
    return;
  float u = ((x + 0.5f) / (float)ow) * 2.0f - 0.5f;
  float v = ((y + 0.5f) / (float)oh) * 2.0f - 0.5f;
  float2 coord = (float2)(u, v);
  float4 ta = read_imagef(a, sampler, coord);
  float4 tb = read_imagef(b, sampler, coord);
  float4 tc = read_imagef(c, sampler, coord);
  int idx = (y * ow + x) * 4;
  oa[idx+0] = (uchar)(ta.x*255.0f+0.5f); oa[idx+1] = (uchar)(ta.y*255.0f+0.5f);
  oa[idx+2] = (uchar)(ta.z*255.0f+0.5f); oa[idx+3] = (uchar)(ta.w*255.0f+0.5f);
  ob[idx+0] = (uchar)(tb.x*255.0f+0.5f); ob[idx+1] = (uchar)(tb.y*255.0f+0.5f);
  ob[idx+2] = (uchar)(tb.z*255.0f+0.5f); ob[idx+3] = (uchar)(tb.w*255.0f+0.5f);
  oc[idx+0] = (uchar)(tc.x*255.0f+0.5f); oc[idx+1] = (uchar)(tc.y*255.0f+0.5f);
  oc[idx+2] = (uchar)(tc.z*255.0f+0.5f); oc[idx+3] = (uchar)(tc.w*255.0f+0.5f);
}
