.type vx_vec_test, @function
.global vx_vec_test
vx_vec_test:
# vector-vector add routine of 32-bit integers
# void vvaddint32(size_t n, const int*x, const int*y, int*z)
# { for (size_t i=0; i<n; i++) { z[i]=x[i]+y[i]; } }
#
# a0 = n, a1 = x, a2 = y, a3 = z
# Non-vector instructions are indented
    vsetvli t0, a0, e32 # Set vector length based on 32-bit vectors
 loop:
    vlw.v v0, (a1)           # Get first vector
      sub a0, a0, t0         # Decrement number done
      slli t0, t0, 2         # Multiply number done by 4 bytes
      add a1, a1, t0         # Bump pointer
    vlw.v v1, (a2)           # Get second vector
      add a2, a2, t0         # Bump pointer
    vadd.vv v2, v0, v1        # Sum vectors
    vsw.v v2, (a3)           # Store result
      add a3, a3, t0         # Bump pointer
      bnez a0, loop         # Loop back 
    vmacc.vv v1, v2, v2
    ret                    # Finished