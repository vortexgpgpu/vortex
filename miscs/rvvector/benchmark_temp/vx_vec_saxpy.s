.type vx_vec_saxpy, @function
.global vx_vec_saxpy
# void
# saxpy(size_t n, const float a, const float *x, float *y)
# {
#   size_t i;
#   for (i=0; i<n; i++)
#     y[i] = a * x[i] + y[i];
# }
#
# register arguments:
#     a0      n
#     fa0     a
#     a1      x
#     a2      y
vx_vec_saxpy:
    vsetvli a4, a0, e32, m8
saxpy:
    vlw.v v0, (a1)
    sub a0, a0, a4
    slli a4, a4, 2
    add a1, a1, a4
    vlw.v v8, (a2)
    vfmacc.vf v8, fa0, v0
    vsw.v v8, (a2)
    add a2, a2, a4
    bnez a0, saxpy
    ret
#vx_vec_saxpy:
#    vsetvli a4, a0, e32, m8
#saxpy:
#    vlw.v v0, (a1)
#    sub a0, a0, a4
#    slli a4, a4, 2
#    add a1, a1, a4
#    vlw.v v8, (a2)
#    vfmacc.vf v8, fa0, v0
#    vsw.v v8, (a2)
#    add a2, a2, a4
#    bnez a0, saxpy
#    ret

# a0 n, rs1 a, a2 x, a3 y

# a0 n, a1 a, a2 x, a3 y
vx_vec_saxpy:
    vsetvli a4, a0, e32, m1
saxpy:
    vlw.v v0, (a2)
    sub a0, a0, a4
    slli a4, a4, 2
    add a2, a2, a4
    vlw.v v1, (a3)
    vmul.vx v0, v0, a1
#    vmul.vv v0, v0, v1
#    li x1, 10
#    vmul.vx v0, v0, x1
    vadd.vv v1, v0, v1
    vsw.v v1, (a3)
    add a3, a3, a4
    bnez a0, saxpy
    ret
