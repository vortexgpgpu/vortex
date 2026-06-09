/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*############################################################################*/

#ifndef _LAYOUT_CONFIG_H_
#define _LAYOUT_CONFIG_H_

/*############################################################################*/

//Unchangeable settings: volume simulation size for the given example
#define SIZE_X (32)
#define SIZE_Y (32)
#define SIZE_Z (8)
//#define SIZE_X (120)
//#define SIZE_Y (120)
//#define SIZE_Z (150)

//Changeable settings
//Padding in each dimension
#define PADDING_X (8)
#define PADDING_Y (0)
#define PADDING_Z (4)

//Pitch in each dimension
#define PADDED_X (SIZE_X+PADDING_X)
#define PADDED_Y (SIZE_Y+PADDING_Y)
#define PADDED_Z (SIZE_Z+PADDING_Z)

#define TOTAL_CELLS (SIZE_X*SIZE_Y*SIZE_Z)
#define TOTAL_PADDED_CELLS (PADDED_X*PADDED_Y*PADDED_Z)

//Flattening function
//  This macro will be used to map a 3-D index and element to a value
//  The macro below implements the equivalent of a 3-D array of 
//  20-element structures in C standard layout.
#define CALC_INDEX(x,y,z,e) ( e + N_CELL_ENTRIES*\
                               ((x)+(y)*PADDED_X+(z)*PADDED_X*PADDED_Y) )

#define MARGIN (CALC_INDEX(0, 0, 2, 0) - CALC_INDEX(0,0,0,0))

// Set this value to 1 for GATHER, or 0 for SCATTER
#if 1
#define GATHER
#else
#define SCATTER
#endif

//OpenCL block size (not trivially changeable here)
#define BLOCK_SIZE SIZE_X

/*############################################################################*/

typedef enum {C = 0,
              N, S, E, W, T, B,
              NE, NW, SE, SW,
              NT, NB, ST, SB,
              ET, EB, WT, WB,
              FLAGS, N_CELL_ENTRIES} CELL_ENTRIES;

#define N_DISTR_FUNCS FLAGS

typedef enum {OBSTACLE    = 1 << 0,
              ACCEL       = 1 << 1,
              IN_OUT_FLOW = 1 << 2} CELL_FLAGS;

#endif /* _CONFIG_H_ */
