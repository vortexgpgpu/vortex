/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef MAX_POS
#define MAX_POS 1089
#define CEIL_POS 61
#define POS_PER_THREAD 18
#define MAX_POS_PADDED 1096
#define THREADS_W 1
#define THREADS_H 1
#define SEARCH_RANGE 16
#define SEARCH_DIMENSION 33
#endif

/* The compute kernel. */
/* The macros THREADS_W and THREADS_H specify the width and height of the
 * area to be processed by one thread, measured in 4-by-4 pixel blocks.
 * Larger numbers mean more computation per thread block.
 *
 * The macro POS_PER_THREAD specifies the number of search positions for which
 * an SAD is computed.  A larger value indicates more computation per thread,
 * and fewer threads per thread block.  It must be a multiple of 3 and also
 * must be at most 33 because the loop to copy from shared memory uses
 * 32 threads per 4-by-4 pixel block.
 *
 */
 
// AMD OpenCL fails UINT_CUDA_V
#define SHORT2_V 0
#define UINT_CUDA_V 0

// Either works
#define VEC_LOAD 0

// CAST_STORE is only method that works for all implementations of OpenCL tested
#define VEC_STORE 0
#define CAST_STORE 0
#define SCALAR_STORE 1

__kernel void mb_sad_calc(__global unsigned short *blk_sad,
                            __global unsigned short *frame,
                            int mb_width,
                            int mb_height,
                            __global unsigned short* img_ref) // __read_only image2d_t img_ref)
{   
  int tx = (get_local_id(0) / CEIL_POS) % THREADS_W;
  int ty = (get_local_id(0) / CEIL_POS) / THREADS_W;
  int bx = get_group_id(0);
  int by = get_group_id(1);
  int img_width = mb_width*16;
  int lidx = get_local_id(0);

  // Macroblock and sub-block coordinates
  int mb_x = (tx + bx * THREADS_W) >> 2;
  int mb_y = (ty + by * THREADS_H) >> 2;
  int block_x = (tx + bx * THREADS_W) & 0x03;
  int block_y = (ty + by * THREADS_H) & 0x03;

  // If this thread is assigned to an invalid 4x4 block, do nothing 
  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      // Pixel offset of the origin of the current 4x4 block
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      // Origin of the search area for this 4x4 block
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      // Origin in the current frame for this 4x4 block
      int cur_o = frame_y * img_width + frame_x;

      int search_pos;
      int search_pos_base =
        (lidx % CEIL_POS) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      // Don't go past bounds
      if (search_pos_end > MAX_POS) {
        search_pos_end = MAX_POS;
      }

      // For each search position, within the range allocated to this thread
      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
        unsigned short sad4x4 = 0;
        int search_off_x = ref_x + (search_pos % SEARCH_DIMENSION);
        int search_off_y = ref_y + (search_pos / SEARCH_DIMENSION);

        // 4x4 SAD computation
        for(int y=0; y<4; y++) {
          for (int x=0; x<4; x++) {
          
          // ([unsigned] short)read_imageui or
          //                   read_imagei  is required for correct calculation.
          // Though read_imagei() is shorter, its results are undefined by specification since the input
          // is an unsigned type, CL_UNSIGNED_INT16
          
            int sx = search_off_x + x;
            sx = (sx < 0) ? 0 : sx;
            sx = (sx >= img_width) ? img_width - 1 : sx;
            int sy = search_off_y + y;
            sy = (sy < 0) ? 0 : sy;
            sy = (sy >= mb_height * 16) ? mb_height * 16 - 1 : sy;
            sad4x4 += abs((unsigned short) img_ref[(sx) + (sy) * img_width]  - 
                  frame[cur_o + y * img_width + x]);
          }
        }

        // Save this value into the local SAD array 
        blk_sad[mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 +
        (4 * block_y + block_x) * MAX_POS_PADDED+search_pos] = sad4x4;
      }
    }

}


//typedef unsigned int uint;

__kernel void larger_sad_calc_8(__global unsigned short *blk_sad,
				  int mb_width,
				  int mb_height)
{
  int tx = get_local_id(1) & 1;
  int ty = get_local_id(1) >> 1;

  // Macroblock and sub-block coordinates
  int mb_x = get_group_id(0);
  int mb_y = get_group_id(1);
  int lidx = get_local_id(0);

  // Number of macroblocks in a frame
  int macroblocks = mul24(mb_width, mb_height);
  int macroblock_index = (mul24(mb_y, mb_width) + mb_x) * MAX_POS_PADDED;

  __global unsigned short *bi;
  __global unsigned short *bo_6, *bo_5, *bo_4;

  // MXPA
  bo_4 = (__global unsigned short *) tx;
  bo_5 = (__global unsigned short *) tx;


  bi = blk_sad    
    + (mul24(macroblocks, 25) + (ty * 8 + tx * 2)) * MAX_POS_PADDED
    + macroblock_index * 16;

  // Block type 6: 4x8
  bo_6 = blk_sad
    + ((macroblocks << 4) + macroblocks + (ty * 4 + tx * 2)) * MAX_POS_PADDED
    + macroblock_index * 8;

  if (ty < 100) // always true, but improves register allocation
    {
      // Block type 5: 8x4
      bo_5 = blk_sad
	+ ((macroblocks << 3) + macroblocks + (ty * 4 + tx)) * MAX_POS_PADDED
	+ macroblock_index * 8;

      // Block type 4: 8x8
      bo_4 = blk_sad
	+ ((macroblocks << 2) + macroblocks + (ty * 2 + tx)) * MAX_POS_PADDED
	+ macroblock_index * 4;
    }

  for (int search_pos = lidx; search_pos < (MAX_POS+1)/2; search_pos += 32)
    {
#if SHORT2_V
  #if VEC_LOAD
      ushort2 s00 = vload2(search_pos,                    bi);
      ushort2 s01 = vload2(search_pos+  MAX_POS_PADDED/2, bi);
      ushort2 s10 = vload2(search_pos+4*MAX_POS_PADDED/2, bi);
      ushort2 s11 = vload2(search_pos+5*MAX_POS_PADDED/2, bi);
  #else
      ushort2 s00 = (ushort2) (bi[search_pos*2], bi[search_pos*2+1]);
      ushort2 s01 = (ushort2) (bi[(search_pos + MAX_POS_PADDED/2)*2], bi[(search_pos + MAX_POS_PADDED/2)*2+1]);
      ushort2 s10 = (ushort2) (bi[(search_pos + 4*MAX_POS_PADDED/2)*2], bi[(search_pos + 4*MAX_POS_PADDED/2)*2+1]);
      ushort2 s11 = (ushort2) (bi[(search_pos + 5*MAX_POS_PADDED/2)*2], bi[(search_pos + 5*MAX_POS_PADDED/2)*2+1]);
  #endif

  #if VEC_STORE
      ushort2 s0010 = s00 + s10;
      ushort2 s0111 = s01 + s11;
      ushort2 s0001 = s00 + s01;
      ushort2 s1011 = s10 + s11;
      ushort2 s00011011 = s0001 + s1011;
      
      vstore2(s0010, search_pos, bo_6);
      vstore2(s0111, search_pos+MAX_POS_PADDED/2, bo_6);
      vstore2(s0001, search_pos, bo_5);
      vstore2(s1011, search_pos+2*MAX_POS_PADDED/2, bo_5);
      vstore2(s00011011, search_pos, bo_4);
  #elif CAST_STORE
      ((__global ushort2 *)bo_6)[search_pos]                  = s00 + s10;
      ((__global ushort2 *)bo_6)[search_pos+MAX_POS_PADDED/2] = s01 + s11;
      ((__global ushort2 *)bo_5)[search_pos]                  = s00 + s01;
      ((__global ushort2 *)bo_5)[search_pos+2*MAX_POS_PADDED/2] = s10 + s11;
      ((__global ushort2 *)bo_4)[search_pos]                  = (s00 + s01) + (s10 + s11);
  #else // SCALAR_STORE
      bo_6[search_pos*2] = s00.x + s10.x;
      bo_6[search_pos*2+1] = s00.y + s10.y;
      bo_6[(search_pos+MAX_POS_PADDED/2)*2] = s01.x + s11.x;
      bo_6[(search_pos+MAX_POS_PADDED/2)*2+1] = s01.y + s11.y;
      bo_5[search_pos*2] = s00.x + s01.x;
      bo_5[search_pos*2+1] = s00.y + s01.y;
      bo_5[(search_pos+2*MAX_POS_PADDED/2)*2] = s10.x + s11.x;
      bo_5[(search_pos+2*MAX_POS_PADDED/2)*2+1] = s10.y + s11.y;
      bo_4[search_pos*2] = (s00.x + s01.x) + (s10.x + s11.x);
      bo_4[search_pos*2+1] = (s00.y + s01.y) + (s10.y + s11.y);
  #endif
#else // UINT_CUDA_V
      uint i00 = ((__global uint *)bi)[search_pos];
      uint i01 = ((__global uint *)bi)[search_pos + MAX_POS_PADDED/2];
      uint i10 = ((__global uint *)bi)[search_pos + 4*MAX_POS_PADDED/2];
      uint i11 = ((__global uint *)bi)[search_pos + 5*MAX_POS_PADDED/2];

      ((__global uint *)bo_6)[search_pos]                  = i00 + i10;
      ((__global uint *)bo_6)[search_pos+MAX_POS_PADDED/2] = i01 + i11;
      ((__global uint *)bo_5)[search_pos]                  = i00 + i01;
      ((__global uint *)bo_5)[search_pos+2*MAX_POS_PADDED/2] = i10 + i11;
      ((__global uint *)bo_4)[search_pos]                  = (i00 + i01) + (i10 + i11);
#endif
    }
    
}



__kernel void larger_sad_calc_16(__global unsigned short *blk_sad,
				   int mb_width,
				   int mb_height)
{
  // Macroblock coordinates 
  int mb_x = get_group_id(0);
  int mb_y = get_group_id(1);
  int search_pos = get_local_id(0);

  // Number of macroblocks in a frame
  int macroblocks = mul24(mb_width, mb_height) * MAX_POS_PADDED;
  int macroblock_index = (mul24(mb_y, mb_width) + mb_x) * MAX_POS_PADDED;

  __global unsigned short *bi;
  __global unsigned short *bo_3, *bo_2, *bo_1;

  //bi = blk_sad + macroblocks * 5 + macroblock_index * 4;
  bi = blk_sad + ((macroblocks + macroblock_index) << 2) + macroblocks;

  // Block type 3: 8x16
  //bo_3 = blk_sad + macroblocks * 3 + macroblock_index * 2;
  bo_3 = blk_sad + ((macroblocks + macroblock_index) << 1) + macroblocks;

  // Block type 5: 8x4
  bo_2 = blk_sad + macroblocks + macroblock_index * 2;

  // Block type 4: 8x8
  bo_1 = blk_sad + macroblock_index;

  for ( ; search_pos < (MAX_POS+1)/2; search_pos += 32)
    {
#if SHORT2_V
  #if VEC_LOAD
      ushort2 s00 = vload2(search_pos,                    bi);
      ushort2 s01 = vload2(search_pos+  MAX_POS_PADDED/2, bi);
      ushort2 s10 = vload2(search_pos+2*MAX_POS_PADDED/2, bi);
      ushort2 s11 = vload2(search_pos+3*MAX_POS_PADDED/2, bi);
  #else
      ushort2 s00 = (ushort2) (bi[search_pos*2], bi[search_pos*2+1]);
      ushort2 s01 = (ushort2) (bi[(search_pos + MAX_POS_PADDED/2)*2], bi[(search_pos + MAX_POS_PADDED/2)*2+1]);
      ushort2 s10 = (ushort2) (bi[(search_pos + 2*MAX_POS_PADDED/2)*2], bi[(search_pos + 2*MAX_POS_PADDED/2)*2+1]);
      ushort2 s11 = (ushort2) (bi[(search_pos + 3*MAX_POS_PADDED/2)*2], bi[(search_pos + 3*MAX_POS_PADDED/2)*2+1]);
  #endif

  #if VEC_STORE
      ushort2 s0010 = s00 + s10;
      ushort2 s0111 = s01 + s11;
      ushort2 s0001 = s00 + s01;
      ushort2 s1011 = s10 + s11;
      ushort2 s00011011 = s0001 + s1011;
      
      vstore2(s0010, search_pos, bo_3);
      vstore2(s0111, search_pos+MAX_POS_PADDED/2, bo_3);
      vstore2(s0001, search_pos, bo_2);
      vstore2(s1011, search_pos+MAX_POS_PADDED/2, bo_2);
      vstore2(s00011011, search_pos, bo_1);
  #elif CAST_STORE
      ((__global ushort2 *)bo_3)[search_pos]                  = s00 + s10;
      ((__global ushort2 *)bo_3)[search_pos+MAX_POS_PADDED/2] = s01 + s11;
      ((__global ushort2 *)bo_2)[search_pos]                  = s00 + s01;
      ((__global ushort2 *)bo_2)[search_pos+MAX_POS_PADDED/2] = s10 + s11;
      ((__global ushort2 *)bo_1)[search_pos]                  = (s00 + s01) + (s10 + s11);
  #else // SCALAR_STORE
      bo_3[search_pos*2] = s00.x + s10.x;
      bo_3[search_pos*2+1] = s00.y + s10.y;
      bo_3[(search_pos+MAX_POS_PADDED/2)*2] = s01.x + s11.x;
      bo_3[(search_pos+MAX_POS_PADDED/2)*2+1] = s01.y + s11.y;
      bo_2[search_pos*2] = s00.x + s01.x;
      bo_2[search_pos*2+1] = s00.y + s01.y;
      bo_2[(search_pos+MAX_POS_PADDED/2)*2] = s10.x + s11.x;
      bo_2[(search_pos+MAX_POS_PADDED/2)*2+1] = s10.y + s11.y;
      bo_1[search_pos*2] = (s00.x + s01.x) + (s10.x + s11.x);
      bo_1[search_pos*2+1] = (s00.y + s01.y) + (s10.y + s11.y);
  #endif
#else // UINT_CUDA_V
      uint i00 = ((__global uint *)bi)[search_pos];
      uint i01 = ((__global uint *)bi)[search_pos + MAX_POS_PADDED/2];
      uint i10 = ((__global uint *)bi)[search_pos + 2*MAX_POS_PADDED/2];
      uint i11 = ((__global uint *)bi)[search_pos + 3*MAX_POS_PADDED/2];

      ((__global uint *)bo_3)[search_pos]                  = i00 + i10;
      ((__global uint *)bo_3)[search_pos+MAX_POS_PADDED/2] = i01 + i11;
      ((__global uint *)bo_2)[search_pos]                  = i00 + i01;
      ((__global uint *)bo_2)[search_pos+MAX_POS_PADDED/2] = i10 + i11;
      ((__global uint *)bo_1)[search_pos]                  = (i00 + i01) + (i10 + i11);
#endif
    }
}


