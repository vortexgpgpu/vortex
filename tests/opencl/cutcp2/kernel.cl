/*
 * potential lattice is decomposed into size 8^3 lattice point "regions"
 *
 * THIS IMPLEMENTATION:  one thread per lattice point
 * thread block size 128 gives 4 thread blocks per region
 * kernel is invoked for each x-y plane of regions,
 * where gridDim.x is 4*(x region dimension) so that blockIdx.x 
 * can absorb the z sub-region index in its 2 lowest order bits
 *
 * Regions are stored contiguously in memory in row-major order
 *
 * The bins have to not only cover the region, but they need to surround
 * the outer edges so that region sides and corners can still use
 * neighbor list stencil.  The binZeroAddr is actually a shifted pointer into
 * the bin array (binZeroAddr = binBaseAddr + (c*binDim_y + c)*binDim_x + c)
 * where c = ceil(cutoff / binsize).  This allows for negative offsets to
 * be added to myBinIndex.
 *
 * The (0,0,0) spatial origin corresponds to lower left corner of both
 * regionZeroAddr and binZeroAddr.  The atom coordinates are translated
 * during binning to enforce this assumption.
 */

#include "macros.h"

// OpenCL 1.1 support for int3 is not uniform on all implementations, so
// we use int4 instead.  Only the 'x', 'y', and 'z' fields of xyz are used.
typedef int4 xyz;

__kernel void opencl_cutoff_potential_lattice(
    int binDim_x,
    int binDim_y,
    __global float4 *binBaseAddr,
    int offset,
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float inv_cutoff2,
    __global float *regionZeroAddr,  /* address of lattice regions starting at origin */
    int zRegionIndex,
    __constant int *NbrListLen,
    __constant xyz *NbrList
    )
{
  __global float4* binZeroAddr = binBaseAddr + offset;

  __global float *myRegionAddr;
  int Bx, By, Bz;

  /* thread id */
  const int tid = (get_local_id(2)*get_local_size(1) + 
                      get_local_id(1))*get_local_size(0) + get_local_id(0);

  /* this is the start of the sub-region indexed by tid */
  myRegionAddr = regionZeroAddr + ((zRegionIndex*get_num_groups(1)
	+ get_group_id(1))*(get_num_groups(0)>>2) + (get_group_id(0)>>2))*REGION_SIZE
	+ (get_group_id(0)&3)*SUB_REGION_SIZE;

  /* spatial coordinate of this lattice point */
  float x = (8 * (get_group_id(0) >> 2) + get_local_id(0)) * h;
  float y = (8 * get_group_id(1) + get_local_id(1)) * h;
  float z = (8 * zRegionIndex + 2*(get_group_id(0)&3) + get_local_id(2)) * h;

  float dx;
  float dy;
  float dz;
  float r2;
  float s;

  int totalbins = 0;

  /* bin number determined by center of region */
  Bx = (int) floor((8 * (get_group_id(0) >> 2) + 4) * h * BIN_INVLEN);
  By = (int) floor((8 * get_group_id(1) + 4) * h * BIN_INVLEN);
  Bz = (int) floor((8 * zRegionIndex + 4) * h * BIN_INVLEN);

  float energy = 0.f;
  int bincnt;
  for (bincnt = 0; bincnt < *NbrListLen;  bincnt++) {
    int i = Bx + NbrList[bincnt].x;
    int j = By + NbrList[bincnt].y;
    int k = Bz + NbrList[bincnt].z;

  	__global float4* p_global = binZeroAddr + 
  	                       (((k*binDim_y + j)*binDim_x + i) * BIN_DEPTH);

    int m;
    for (m = 0;  m < BIN_DEPTH;  m++) {
    	float aq = p_global[m].w;
      if (0.f != aq) {
        dx = p_global[m].x - x;
        dy = p_global[m].y - y;
        dz = p_global[m].z - z;
        r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < cutoff2) {
          s = (1.f - r2 * inv_cutoff2);
          energy += aq * rsqrt(r2) * s * s;
        }
      }
    } /* end loop over atoms in bin */
  } /* end loop over neighbor list */

  /* store into global memory */
  myRegionAddr[tid+0] = energy;
}
