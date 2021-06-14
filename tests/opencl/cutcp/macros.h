#ifndef __MACROSH__
#define __MACROSH__

#ifdef __DEVICE_EMULATION__
#define DEBUG
/* define which grid block and which thread to examine */
#define BX  0
#define BY  0
#define TX  0
#define TY  0
#define TZ  0
#define EMU(code) do { \
  if (blockIdx.x==BX && blockIdx.y==BY && \
      threadIdx.x==TX && threadIdx.y==TY && threadIdx.z==TZ) { \
    code; \
  } \
} while (0)
#define INT(n)    printf("%s = %d\n", #n, n)
#define FLOAT(f)  printf("%s = %g\n", #f, (double)(f))
#define INT3(n)   printf("%s = %d %d %d\n", #n, (n).x, (n).y, (n).z)
#define FLOAT4(f) printf("%s = %g %g %g %g\n", #f, (double)(f).x, \
    (double)(f).y, (double)(f).z, (double)(f).w)
#else
#define EMU(code)
#define INT(n)
#define FLOAT(f)
#define INT3(n)
#define FLOAT4(f)
#endif

/* report error from OpenCL */
#define CHECK_ERROR(errorMessage)           \
  if(clStatus != CL_SUCCESS)                \
  {                                         \
     printf("Error: %s!\n",errorMessage);   \
     printf("Line: %d\n",__LINE__);         \
     exit(1);                               \
  }

/*
 * neighbor list:
 * stored in constant memory as table of offsets
 * flat index addressing is computed by kernel
 *
 * reserve enough memory for 11^3 stencil of grid cells
 * this fits within 16K of memory
 */
#define NBRLIST_DIM  11
#define NBRLIST_MAXLEN (NBRLIST_DIM * NBRLIST_DIM * NBRLIST_DIM)

/*
 * atom bins cached into shared memory for processing
 *
 * this reserves 4K of shared memory for 32 atom bins each containing 8 atoms,
 * should permit scheduling of up to 3 thread blocks per SM
 */
#define BIN_DEPTH         8  /* max number of atoms per bin */
#define BIN_SIZE         32  /* size of bin in floats */
#define BIN_CACHE_MAXLEN 32  /* max number of atom bins to cache */

#define BIN_LENGTH      4.f  /* spatial length in Angstroms */
#define BIN_INVLEN  (1.f / BIN_LENGTH)
/* assuming density of 1 atom / 10 A^3, expectation is 6.4 atoms per bin
 * so that bin fill should be 80% (for non-empty regions of space) */

#define REGION_SIZE     512  /* number of floats in lattice region */
#define SUB_REGION_SIZE 128  /* number of floats in lattice sub-region */

#endif
