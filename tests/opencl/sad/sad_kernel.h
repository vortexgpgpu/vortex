/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* Integer ceiling division.  This computes ceil(x / y) */
#define CEIL(x,y) (((x) + ((y) - 1)) / (y))

/* Fast multiplication by 33 */
#define TIMES_DIM_POS(x) (((x) << 5) + (x))

/* Amount of dynamically allocated local storage
 * measured in bytes, 2-byte words, and 8-byte words */
#define SAD_LOC_SIZE_ELEMS (THREADS_W * THREADS_H * MAX_POS_PADDED)
#define SAD_LOC_SIZE_BYTES (SAD_LOC_SIZE_ELEMS * sizeof(unsigned short))
#define SAD_LOC_SIZE_8B (SAD_LOC_SIZE_BYTES / sizeof(vec8b))

/* The search position index space is distributed across threads
 * and across time. */
/* This many search positions are calculated by each thread.
 * Note: the optimized kernel requires that this number is
 * divisible by 3. */
#define POS_PER_THREAD 18

/* The width and height (in number of 4x4 blocks) of a tile from the
 * current frame that is computed in a single thread block. */
#define THREADS_W 1
#define THREADS_H 1

// #define TIMES_THREADS_W(x) (((x) << 1) + (x))
#define TIMES_THREADS_W(x) ((x) * THREADS_W)

/* This structure is used for vector load/store operations. */

struct vec8b {
  int fst;
  int snd;
} __attribute__ ((aligned(8)));



/* 4-by-4 SAD computation on the device. */
/*
extern "C" __global__ void mb_sad_calc(unsigned short*,
			    unsigned short*,
			    int, int);
*/
/* A function to get a reference to the "ref" texture, because sharing
 * of textures between files isn't really supported. */
 /*
texture<unsigned short, 2, cudaReadModeElementType> &get_ref(void);

extern "C" __global__ void larger_sad_calc_8(unsigned short*, int, int);
extern "C" __global__ void larger_sad_calc_16(unsigned short*, int, int);*/
