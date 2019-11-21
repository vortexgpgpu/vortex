


.type vx_vec_test, @function
.global vx_vec_test
vx_vec_test:
	li a0, 2
	vsetvli t0, a0, e32, m1
	li  a0, 10
	sw  a0, 0(a1)
	sw  a0, 32(a1)
	vlw.v   v0, (a1)
	vadd.vv v0, v0, v0
	vsw.v  v0, (a1)
	ret
