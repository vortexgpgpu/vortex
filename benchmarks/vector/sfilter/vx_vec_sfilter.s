.type vx_vec_saxpy, @function
.global vx_vec_sfilter
#vx_vec_sfilter(a, b, ldc, m, x, y, N);
#a0 - a    
#a1 - b
#a2 - ldc
#a3 - m
#a4 - x
#a5 - y
#a6 - N

vx_vec_sfilter:
    vsetvli t0, a6, e32

    li t1, 1
    sub t2, a4, t1 #(x-1)
    add t3, a4, t1 #(x+1)
    sub t4, a5, t1 #(y-1)
    add t5, a5, t1 #(y+1)

    #i0
    mul t6, t4, a2 #(y-1)*ldc
    add a7, t6, t2 #(x-1) + (y-1)*ldc
    slli a7, a7, 2
    add a0, a0, a7
    vlw.v v0, (a0)
    vmul.vx v0, v0, a3
    sub a0, a0, a7

    #i1
    add a7, t6, a4 #(x + (y-1)*ldc)
    slli a7, a7, 2
    add a0, a0, a7
    vlw.v v1, (a0)
    vmul.vx v1, v1, a3
    sub a0, a0, a7

    #i2
    add a7, t3, t6 #((x+1) + (y-1)*ldc)
    slli a7, a7, 2
    add a0, a0, a7
    vlw.v v2, (a0)
    vmul.vx v2, v2, a3
    sub a0, a0, a7

    #i3
    mul t6, a5, a2 #y*ldc
    add a7, t6, t2 #(x-1) + y*ldc
    slli a7, a7, 2
    add a0, a0, a7
    vlw.v v3, (a0)
    vmul.vx v3, v3, a3
    sub a0, a0, a7

    #i4
    add a7, t6, a4 #(x + y*ldc)
    slli a7, a7, 2
    add a0, a0, a7
    vlw.v v4, (a0)
    vmul.vx v4, v4, a3
    sub a0, a0, a7

    #i5
    add a7, t6, t3 #((x+1) + (y*ldc))
    slli a7, a7, 2
    add a0, a0, a7
    vlw.v v5, (a0)
    vmul.vx v5, v5, a3
    sub a0, a0, a7

    #i6
    mul t6, t5, a2 #(y+1)*ldc
    add a7, t6, t2 #(x-1) + (y+1)*ldc
    slli a7, a7, 2
    add a0, a0, a7
    vlw.v v6, (a0)
    vmul.vx v6, v6, a3
    sub a0, a0, a7

    #i7
    add a7, t6, a4 #(y+1)*ldc + x
    slli a7, a7, 2
    add a0, a0, a7
    vlw.v v7, (a0)
    vmul.vx v7, v7, a3
    sub a0, a0, a7

    #i8
    add a7, t6, t3 #(x+1) + (y+1)*ldc
    slli a7, a7, 2
    add a0, a0, a7
    vlw.v v8, (a0)
    vmul.vx v8, v8, a3
    sub a0, a0, a7

    #c
    mul t6, a5, a2 #y*ldc
    add a7, t6, a4 # x + y*ldc
    vadd.vv v9, v0, v1
    vadd.vv v9, v9, v2
    vadd.vv v9, v9, v3
    vadd.vv v9, v9, v4
    vadd.vv v9, v9, v5
    vadd.vv v9, v9, v6
    vadd.vv v9, v9, v7
    vadd.vv v9, v9, v8
    slli a7, a7, 2
    add a1, a1, a7
    vsw.v v9, (a1)

    ret
