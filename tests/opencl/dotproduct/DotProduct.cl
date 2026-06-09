/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 __kernel void DotProduct (__global float* a, __global float* b, __global float* c, int iNumElements)
{
    // find position in global arrays
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= iNumElements)
    {   
        return; 
    }

    // process 
    int iInOffset = iGID << 2;
    c[iGID] = a[iInOffset] * b[iInOffset] 
               + a[iInOffset + 1] * b[iInOffset + 1]
               + a[iInOffset + 2] * b[iInOffset + 2]
               + a[iInOffset + 3] * b[iInOffset + 3];
}
