// Validates the get_image_* query builtins by writing them to a buffer.
__kernel void image_get(read_only image2d_t img, __global int* out) {
  if (get_global_id(0) == 0 && get_global_id(1) == 0) {
    out[0] = get_image_width(img);
    out[1] = get_image_height(img);
    out[2] = get_image_channel_data_type(img);
    out[3] = get_image_channel_order(img);
  }
}
