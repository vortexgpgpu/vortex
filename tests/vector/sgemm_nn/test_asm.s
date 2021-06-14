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
# a3 = a, a4 = b, a5 = c
# a0 = i, a1 = m, a2 = n
# a6 = ldc
vx_vec_sgemm_nn:
    vsetvli t0, a6, e32
    mul x1, a6, a2  # n*ldc
    add x2, x1, a1  # i + (n*ldc)
    add a3, x2, a3  # a[i+ n*ldc]
    lw x3, (a3)

    mul x4, a1, a6  # m*ldc
    add x5, a0, x4  # i + m*ldc
    add a4, x5, a4  # b[i + m*ldc]
#   lw x6, (a4)

    vlw.v v0, (a4)
    vmul.vx v2, v1, x3
 
    mul x6, a2, a6  # n*ldc
    add x7, a0, x6  # i + n*ldc
    add a5, x7, a5  # c[i + m*ldc]

    vlw.v v3, (a5) #c
    vadd.vv v3, v3, v2

    ret
