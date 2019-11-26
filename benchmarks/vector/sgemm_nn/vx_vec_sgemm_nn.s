.type vx_vec_sgemm_nn, @function
.global vx_vec_sgemm_nn
#
#  for (int n = 0; n < k; n++) {
#       for (int m = 0; m < m; m++) {
#           for (int i = 0; i < n;) {
#//               d1[n*k+i] += a1[n*k+m]*b1[i*n+m];
#                 vx_vec_sgemm_nn(i, c, r, a1, b1, c1, ldc);
#                 i = i + 4;
#           }
#       }
#    } 
# a0 = i, a1 = m, a2 = n,  a3 = a, a4 = b, a5 = c, a6 = ldc, a7 = vsize
#
vx_vec_sgemm_nn:
    vsetvli t0, a7, e32 # <--- vsize 
    mul x11, a6, a2  # n*ldc
    add x12, x11, a1  # i + (n*ldc)
    slli x12, x12, 2 
    add a3, x12, a3  # a[i+ n*ldc]
    lw x13, (a3)

    mul x14, a1, a6  # m*ldc
    add x15, a0, x14  # i + m*ldc
    slli x15, x15, 2
    add a4, x15, a4  # b[i + m*ldc]
    vlw.v v0, (a4)
    vmul.vx v2, v1, x13
##   lw x6, (a4)
##   lw x10, (a4)  # b
##   mul x11, x3, x10

    mul x6, a2, a6  # n*ldc
    add x7, a0, x6  # i + n*ldc
    add a5, x7, a5  # c[i + m*ldc]
    vlw.v v3, (a5) # c
    vadd.vv v3, v3, v2
    vsw.v v3, (a5)
##   lw x12, (a5)
##   add x12, x12, x11
##   sw x12, (a5)    
    ret
