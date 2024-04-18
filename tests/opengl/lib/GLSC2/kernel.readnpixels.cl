__kernel void gl_rgba4_rgba8(
    __global const unsigned short* buf_in,
    __global unsigned int* buf_out,
    const int x,
    const int y,
    const int width,
    const int height
) {
  int gid = get_global_id(0);
  // cordinates of the (i,j)
  int i = gid % width;
  int j = gid / width;

  unsigned short color4 = buf_in[(j+y)*width+(i+x)];

  unsigned int color8 = 0x0;
  color8 |= (unsigned int) (color4 & 0x000F) << 4 | (color4 & 0x000F);
  color8 |= (unsigned int) (color4 & 0x00F0) << 8 | (color4 & 0x00F0) << 4;
  color8 |= (unsigned int) (color4 & 0x0F00) << 12 | (color4 & 0x0F00) << 8;
  color8 |= (unsigned int) (color4 & 0xF000) << 16 | (color4 & 0xF000) << 12;

  buf_out[(height-j-1)*width+i] = color8;
}
