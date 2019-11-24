.type vx_vec_sgemm_nn, @function
.global vx_vec_sgemm_nn
# RV64IDV system
#
# void
# sgemm_nn(size_t n, size_t m, size_t k,
#          int *a,   // m * k matrix
#          int *b,   // k * n matrix
#          int *c)   // m * n matrix
#
#  c += a*b (alpha=1, no transpose on input matrices)
#  matrices stored in C row-major order
#
# for (int r = 0; r < k; r++) {
#    for (int c = 0; c < m; c++) {
#        for (int i = 0; i < n; i++) {
#            c[r*k+i] += a[r*k+c]*b[i*n+c];
#        }
#    }
# }
# a0 = n, a1 = m, a2 = k 
# a3 = a, a4 = b, a5 = c
# v0 = a, v2 = b, v2 = c
# x0 = i, x1 = c, x2 = r 
#
vx_vec_sgemm_nn:
        vsetvli t0, a2, e32, m8 # k
loop_row:                   # a[m][k]
    vlw.v v0, (a3)
        sub a2, a2, t0
        slli t0, t0, 2
        add a3, a3, t0

        vsetvli t1, a1, e32, m8 # m
loop_col:                   # b[k][n]
    vlw.v v1, (a4)
        sub a1, a1, t1
        slli t1, t1, 2
        add a4, a4, t1

        vsetvli t2, a0, e32, m8 # n
loop_iner:
    vlw.v v2, (a5)         # c[][]
        sub a0, a0, t2
        slli t2, t2, 2
        add a5, a5, t2

    bnez t2, loop_iner

    bnez t1, loop_col


 #   vadd.vv v0, v0, v0
 #   vsw.v v0, (a5)
 #   add a5, a5, t0

    bnez t0, loop_row
    ret



