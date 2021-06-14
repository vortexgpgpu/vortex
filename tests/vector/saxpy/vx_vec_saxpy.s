.type vx_vec_saxpy, @function
.global vx_vec_saxpy
# void
# saxpy(size_t n, int factor, int *a, int *b)
# {  for (int i=0; i<n; i++) { y[i] = a * x[i] + y[i];}  }
#
# register arguments:
#     a0      n
#     a1      factor
#     a2      a
#     a3      b
vx_vec_saxpy:
loop:
    vsetvli a4, a0, e32
    vlw.v v0, (a2)
    sub a0, a0, a4
    slli a4, a4, 2
    add a2, a2, a4
    vlw.v v1, (a3)
    vmul.vx v0, v0, a1
    vadd.vv v1, v0, v1
#   vmacc.vx v1, rs1, v0
    vsw.v v1, (a3)
    add a3, a3, a4
    bnez a0, loop
    ret
