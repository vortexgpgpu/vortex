.type vx_mat_muli32, @function
.global vx_mat_mulint32
# vector-vector add routine of 32-bit integers
# void vvaddint32(size_t n, const int*x, const int*y, int*z)
# { for (size_t i=0; i<n; i++) { z[i]=x[i]+y[i]; } }
#
# a0 = C, a1 = A, a2 = B
# Non-vector instructions are indented
vx_mat_mulint32:
    #load from a1 to r1
    #mla
    # #load from a1+1 to r2
    # mla s2, (a1)
    # #load from a1+2 to r3
    # mla s3, (a1)
    # #load from a1+3 to r4
    # lw s4, (a1)

    # #load from a2 to r5
    # lw s5, (a2)
    # #load from a2+1 to r6
    # lw s6, (a2)
    # #load from a2+2 to r7
    # lw s7, (a2)
    # #load from a2+3 to r8
    # lw s8, (a2)

    # #multiply and store in regs t1, t2, t3, t4 

    # #store r9 in a0
    # sw t1, (a0)
    # #store r10 in a0+1
    # sw t2, (a0)
    # #store r11 in a0+2
    # sw t3, (a0)
    # #store r12 in a0+3
    # sw t4, (a0)

    #return
    ret

#loop:
#    vlw.v v0, (a1)           # Get first vector
#      sub a0, a0, t0         # Decrement number done
#      slli t0, t0, 2         # Multiply number done by 4 bytes
#      add a1, a1, t0         # Bump pointer
#    vlw.v v1, (a2)           # Get second vector
#      add a2, a2, t0         # Bump pointer
#    vadd.vv v2, v0, v1        # Sum vectors
#    vsw.v v2, (a3)           # Store result
#      add a3, a3, t0         # Bump pointer
#      bnez a0, loop   # Loop back 
#    ret                    # Finished
