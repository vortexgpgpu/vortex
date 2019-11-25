


.type vx_vec_test, @function
.global vx_vec_test
vx_vec_test:
	li a1, 7
	sw a1, 0(a0)
	ret




# 	slli a0, a0, 2
# 	add a0, a0, a3
# 	vmv.v.x vv0, a2
# 	# vsplat4 vv0, a2
# stripmine_loop:
# 	vlb4 vv1, (a1)
# 	vcmpez4 vp0, vv1
# 	!vp0 vlw4 vv1, (a3)
# 	!vp0 vlw4 vv2, (a4)
# 	!vp0 vfma4 vv1, vv0, vv1, vv2
# 	!vp0 vsw4 vv1, (a4)
# 	addi a1, a1, 4
# 	addi a3, a3, 16
# 	addi a4, a4, 16
# 	bleu a3, a0, stripmine_loop
	# handle edge cases
	# when (n % 4) != 0 ...
