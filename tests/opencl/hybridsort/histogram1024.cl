/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */



////////////////////////////////////////////////////////////////////////////////
// Common definition
////////////////////////////////////////////////////////////////////////////////
//Total number of possible data values
#define BIN_COUNT (1024) // Changed from 256
#define HISTOGRAM_SIZE (BIN_COUNT * sizeof(unsigned int))
//Machine warp size
#ifndef __DEVICE_EMULATION__
//G80's warp size is 32 threads
#define WARP_LOG_SIZE 5
#else
//Emulation currently doesn't execute threads in coherent groups of 32 threads,
//which effectively means warp size of 1 thread for emulation modes
#define WARP_LOG_SIZE 0
#endif
//Warps in thread block
#define  WARP_N 3
//Per-block number of elements in histograms
#define BLOCK_MEMORY (WARP_N * BIN_COUNT)
#define IMUL(a, b) mul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute per-workgroup partial histograms
////////////////////////////////////////////////////////////////////////////////

inline void addData1024(volatile __local uint *s_WarpHist, uint data, uint tag){
    uint count;
    do{
        count = s_WarpHist[data]  & 0x07FFFFFFU;
        count = tag | (count + 1);
        s_WarpHist[data] = count;
    }while(s_WarpHist[data] != count);
}

 __kernel void histogram1024Kernel(
                  __global uint *d_Result,
                  __global float *d_Data,
                     float minimum,
                     float maximum,
                    uint dataCount
                  ){
    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);
    //Per-warp substorage storage
     int mulBase = (get_local_id(0) >> WARP_LOG_SIZE);
     const int warpBase = IMUL(mulBase, BIN_COUNT);
    __local unsigned int s_Hist[BLOCK_MEMORY];
     int test = 0;
     
//     if(get_global_id(0) == 0) {
//     for(int i = 0; i < 1024; i++) {
//             d_Result[i] = 0;
//         }
//     }
     const uint tag =  get_local_id(0) << (32 - WARP_LOG_SIZE);
    //Clear shared memory storage for current threadblock before processing
     for(uint i = get_local_id(0); i < BLOCK_MEMORY; i+=get_local_size(0)){
        s_Hist[i] = 0;
 }

    
    //Read through the entire input buffer, build per-warp histograms
     barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for(int pos = get_global_id(0); pos < dataCount; pos += get_global_size(0)){
        uint data4 = ((d_Data[pos] - minimum)/(maximum - minimum)) * BIN_COUNT;
        addData1024(s_Hist + warpBase, data4 & 0x3FFU, tag);
    }
    
    //Per-block histogram reduction
     // Sum is adding to index 0, pls fix
     barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for(int pos = get_local_id(0); pos < BIN_COUNT; pos += get_local_size(0)){
        uint sum = 0;
        for(int i = 0; i < BLOCK_MEMORY; i+= BIN_COUNT){
            sum += s_Hist[pos + i] & 0x07FFFFFFU;
        }
        atomic_add(d_Result+pos,sum);
    }
     
     
     
}


