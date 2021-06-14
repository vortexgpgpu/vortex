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
    vsetvli t0, a7, e32
     mul t1, a6, a2  # n*ldc
     add t2, t1, a1  # i + (n*ldc)
     slli t2, t2, 2
     add a3, t2, a3  # a[i+ n*ldc]
     lw t3, (a3)

     mul t4, a1, a6  # m*ldc
     add t5, a0, t4  # i + m*ldc
     slli t5, t5, 2
     add a4, t5, a4  # b[i + m*ldc]
 #   lw x6, (a4)

     vlw.v v0, (a4)
     vmul.vx v1, v0, t3
 
     mul t6, a2, a6  # n*ldc
     add t0, a0, t6  # i + n*ldc
     slli t0, t0, 2
     add a5, t0, a5  # c[i + m*ldc]

     vlw.v v2, (a5) #c
     vadd.vv v2, v2, v1
     vsw.v v2, (a5)

    ret
