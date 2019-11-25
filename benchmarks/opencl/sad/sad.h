/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* Search offsets within 16 pixels of (0,0) */
#define SEARCH_RANGE 16

/* The total search area is 33 pixels square */
#define SEARCH_DIMENSION (2*SEARCH_RANGE+1)

/* The total number of search positions is 33^2 */
#define MAX_POS 1089

/* This is padded to a multiple of 8 when allocating memory */
#define MAX_POS_PADDED 1096

/* VBSME block indices in the SAD array for different 
 * block sizes.  The index is computed from the
 * image size in macroblocks.  Block sizes are (height, width):
 *  1: 16 by 16 pixels, one block per macroblock
 *  2: 8  by 16 pixels, 2  blocks per macroblock
 *  3: 16 by 8  pixels, 2  blocks per macroblock
 *  4: 8  by 8  pixels, 4  blocks per macroblock
 *  5: 4  by 8  pixels, 8  blocks per macroblock
 *  6: 8  by 4  pixels, 8  blocks per macroblock
 *  7: 4  by 4  pixels, 16 blocks per macroblock
 */
#define SAD_TYPE_1_IX(image_size) 0
#define SAD_TYPE_2_IX(image_size) ((image_size)*MAX_POS_PADDED)
#define SAD_TYPE_3_IX(image_size) ((image_size)*(3*MAX_POS_PADDED))
#define SAD_TYPE_4_IX(image_size) ((image_size)*(5*MAX_POS_PADDED))
#define SAD_TYPE_5_IX(image_size) ((image_size)*(9*MAX_POS_PADDED))
#define SAD_TYPE_6_IX(image_size) ((image_size)*(17*MAX_POS_PADDED))
#define SAD_TYPE_7_IX(image_size) ((image_size)*(25*MAX_POS_PADDED))

#define SAD_TYPE_IX(n, image_size) \
  ((n == 1) ? SAD_TYPE_1_IX(image_size) : \
   ((n == 2) ? SAD_TYPE_2_IX(image_size) : \
    ((n == 3) ? SAD_TYPE_3_IX(image_size) : \
     ((n == 4) ? SAD_TYPE_4_IX(image_size) : \
      ((n == 5) ? SAD_TYPE_5_IX(image_size) : \
       ((n == 6) ? SAD_TYPE_6_IX(image_size) : \
        (SAD_TYPE_7_IX(image_size) \
	 )))))))

#define SAD_TYPE_1_CT 1
#define SAD_TYPE_2_CT 2
#define SAD_TYPE_3_CT 2
#define SAD_TYPE_4_CT 4
#define SAD_TYPE_5_CT 8
#define SAD_TYPE_6_CT 8
#define SAD_TYPE_7_CT 16

#define SAD_TYPE_CT(n) \
  ((n == 1) ? SAD_TYPE_1_CT : \
   ((n == 2) ? SAD_TYPE_2_CT : \
    ((n == 3) ? SAD_TYPE_3_CT : \
     ((n == 4) ? SAD_TYPE_4_CT : \
      ((n == 5) ? SAD_TYPE_5_CT : \
       ((n == 6) ? SAD_TYPE_6_CT : \
        (SAD_TYPE_7_CT \
	 )))))))

#ifdef __cplusplus
extern "C" {
#endif

void sad4_cpu(unsigned short *blk_sad,
	      unsigned short *frame,
	      unsigned short *ref,
	      int mb_width,
	      int mb_height);

void larger_sads(unsigned short *sads,
		 int mbs);

#ifdef __cplusplus
}
#endif
