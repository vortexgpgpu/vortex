// Identity image copy: read each texel with a nearest, unnormalized,
// clamp-to-edge sampler passed as a kernel argument, and write it to the
// output image. Integer coordinates address the exact texel, so a UNORM_INT8
// round-trip is bit-exact.
__kernel void image_copy(read_only image2d_t src,
                         write_only image2d_t dst,
                         sampler_t sampler) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  if (coord.x >= get_image_width(dst) || coord.y >= get_image_height(dst))
    return;
  float4 texel = read_imagef(src, sampler, coord);
  write_imagef(dst, coord, texel);
}
