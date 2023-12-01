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
 
// OpenCL Kernel Function Naive Implementation for hyptenuse
__kernel void VectorHypot(__global float4* fg4A, __global float4* fg4B, __global float4* fg4Hypot, unsigned int uiOffset, int iInnerLoopCount, unsigned int uiNumElements)
{
    // get index into global data array
    size_t szGlobalOffset = get_global_id(0) + uiOffset;

    // bound check 
    if (szGlobalOffset >= uiNumElements)
    {   
        return; 
    }

    // Processing 4 elements per work item, so read fgA and fgB source values from GMEM
    float4 f4A = fg4A[szGlobalOffset];
    float4 f4B = fg4B[szGlobalOffset];
    float4 f4H = (float4)0.0f;
     
    // Get the hypotenuses the vectors of 'legs', but exaggerate the time needed with loop  
    for (int i = 0; i < iInnerLoopCount; i++)  
    {
        // compute the 4 hypotenuses using built-in function
        f4H.x = hypot (f4A.x, f4B.x);
        f4H.y = hypot (f4A.y, f4B.y);
        f4H.z = hypot (f4A.z, f4B.z);
        f4H.w = hypot (f4A.w, f4B.w);
    }
    
    // Write 4 result values back out to GMEM
    fg4Hypot[szGlobalOffset] = f4H;
}