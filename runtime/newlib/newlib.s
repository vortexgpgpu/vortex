


# .type __lttf2, @function
# .global __lttf2
# __lttf2:
# 	ret


# .type __extenddftf2, @function
# .global __extenddftf2
# __extenddftf2:
# 	ret


# .type __trunctfdf2, @function
# .global __trunctfdf2
# __trunctfdf2:
# 	ret


# .type __multf3, @function
# .global __multf3
# __multf3:
# 	ret


# .type __fixtfsi, @function
# .global __fixtfsi
# __fixtfsi:
# 	ret

# .type __floatsitf, @function
# .global __floatsitf
# __floatsitf:
# 	ret

# .type __subtf3, @function
# .global __subtf3
# __subtf3:
# 	ret

# .type __gttf2, @function
# .global __gttf2
# __gttf2:
# 	ret


# .type __eqtf2, @function
# .global __eqtf2
# __eqtf2:
# 	ret

# .type __netf2, @function
# .global __netf2
# __netf2:
# 	ret

# define __udivdi3 __udivsi3
# define __umoddi3 __umodsi3
# define __divdi3 __divsi3
# define __moddi3 __modsi3

# .type __udivsi3, @function
# .global __udivsi3
# __udivsi3:
#   move   t0, ra
#   jal    __udivdi3
#   jr     t0


# .type __umodsi3, @function
# .global __umodsi3
# __umodsi3:
#   move   t0, ra
#   jal    __udivdi3
#   mv     a0, a1
#   jr     t0



# .type __divsi3, @function
# .global __divsi3
# __divsi3:
#   ret
# #endif

# .type __divdi3, @function
# .global __divdi3
# __divdi3:
#   bltz  a0, .L10
#   bltz  a1, .L11
#   /* Since the quotient is positive, fall into __udivdi3.  */


# .type __udivdi3, @function
# .global __udivdi3
# __udivdi3:
#   mv    a2, a1
#   mv    a1, a0
#   li    a0, -1
#   beqz  a2, .L5
#   li    a3, 1
#   bgeu  a2, a1, .L2
# .L1:
#   blez  a2, .L2
#   slli  a2, a2, 1
#   slli  a3, a3, 1
#   bgtu  a1, a2, .L1
# .L2:
#   li    a0, 0
# .L3:
#   bltu  a1, a2, .L4
#   sub   a1, a1, a2
#   or    a0, a0, a3
# .L4:
#   srli  a3, a3, 1
#   srli  a2, a2, 1
#   bnez  a3, .L3
# .L5:
#   ret

# .type __umoddi3, @function
# .global __umoddi3
# __umoddi3:
#   move  t0, ra
#   jal   __udivdi3
#   move  a0, a1
#   jr    t0

# .L10:
#   neg   a0, a0
#   bgez  a1, .L12      /* Compute __udivdi3(-a0, a1), then negate the result.  */
#   neg   a1, a1
#   j     __udivdi3     /* Compute __udivdi3(-a0, -a1).  */
# .L11:                 /* Compute __udivdi3(a0, -a1), then negate the result.  */
#   neg   a1, a1
# .L12:
#   move  t0, ra
#   jal   __udivdi3
#   neg   a0, a0
#   jr    t0


# .type __moddi3, @function
# .global __moddi3
# __moddi3:
#   move   t0, ra
#   bltz   a1, .L31
#   bltz   a0, .L32
# .L30:
#   jal    __udivdi3    /* The dividend is not negative.  */
#   move   a0, a1
#   jr     t0
# .L31:
#   neg    a1, a1
#   bgez   a0, .L30
# .L32:
#   neg    a0, a0
#   jal    __udivdi3    /* The dividend is hella negative.  */
#   neg    a0, a1
#   jr     t0

# .type __modsi3, @function
# .global __modsi3
# __modsi3:
#   move   t0, ra
#   bltz   a1, .L34
#   bltz   a0, .L35
# .L33:
#   jal    __udivdi3    /* The dividend is not negative.  */
#   move   a0, a1
#   jr     t0
# .L34:
#   neg    a1, a1
#   bgez   a0, .L30
# .L35:
#   neg    a0, a0
#   jal    __udivdi3    /* The dividend is hella negative.  */
#   neg    a0, a1
#   jr     t0

