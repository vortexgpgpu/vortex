


.type vx_vec_test, @function
.global vx_vec_test
vx_vec_test:
	vsetvli t0, x0, e32
	vadd.vv v0, v0, v0
	ret
