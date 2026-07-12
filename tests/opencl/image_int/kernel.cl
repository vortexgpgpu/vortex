// Integer image identity copy: read_imageui + write_imageui over a
// CL_UNSIGNED_INT8 RGBA image (nearest, unnormalized). Exercises the unfiltered
// integer image path.
__kernel void image_int(read_only image2d_t src, write_only image2d_t dst,
                        sampler_t sampler) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  uint4 px = read_imageui(src, sampler, coord);
  write_imageui(dst, coord, px);
}
