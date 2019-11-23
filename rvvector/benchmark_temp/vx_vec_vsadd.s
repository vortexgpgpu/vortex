.type vx_vec_vsadd, @function
.global vx_vec_vsadd
#  vector-scalar add
#  N = a0, a[] = a1, B = a2
#  for (i=0; i<N; i++)
#     { C[i] = A[i] + B; } // 32-bit ints
#
vx_vec_vsadd:
#	vcfg2*V32bINT, 1*S32bINT #
#     vmv v2, a2   # Copy B to vector unit scalar
loop:
#     setvl t0, a0 # a0 holds N, t0 holds amount done
     ld v0, a1    # load strip of A vector
     vadd v1, v0, v2 # add vectors
     st v1, a3    # store strip of C vector
     slli t1, t0, 2 # multiply by 4 to get bytes  
     add a1, a1, t1 # bump pointers  
     add a3, a3, t1  
     sub a0, a0, t0 # Subtract amount done
     bnez a0, loop
