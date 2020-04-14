/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

__kernel void spmv_jds_naive(__global float *dst_vector, __global float *d_data,
                 __global int *d_index, __global int *d_perm,
                 __global float *x_vec, const int dim,
                 __constant int *jds_ptr_int,
                 __constant int *sh_zcnt_int)
{
    int ix = get_global_id(0);

    if (ix < dim) {
        float sum = 0.0f;
        // 32 is warp size
        int bound=sh_zcnt_int[ix/32];

        for(int k=0;k<bound;k++)
        {
            int j = jds_ptr_int[k] + ix;
            int in = d_index[j];

            float d = d_data[j];
            float t = x_vec[in];

            sum += d*t;
        }

        dst_vector[d_perm[ix]] = sum;
     }
}

