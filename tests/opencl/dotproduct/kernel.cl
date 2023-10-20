__kernel void DotProduct (__global float* a, __global float* b, __global float* c, int iNumElements)
{
    // find position in global arrays
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    //printf("%d, %d\n", iGID, iNumElements);
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
    //float cc = c[iGID];

    //printf("c[%d]=%f\n", iGID, cc);
}