


.type vx_vec_test, @function
.global vx_vec_test
vx_vec_test:
	li a0, 2
	vsetvli t0, a0, e32, m1
	li  a0, 10
	sw  a0, 0(a1)
	sw  a0, 32(a1)
	vlw.v   v1, (a1)
	li a2, 1
	sw a2, 0(a3)
	li a2, 0
	sw a2, 32(a3)
	vlw.v v0, (a3)
	vmor.mm v0, v0, v3
	vadd.vv v1, v1, v1, v0.t
	vsw.v  v1, (a1)
	vlw.v  v5, (a1)
	ret
