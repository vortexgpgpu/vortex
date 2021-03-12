#include <VX_config.h>

@ .type vx_tex_ld, @function
@ .global vx_tex_ld
@ vx_tex_ld:
@     slli	a1,a1,0x8
@     or	a1,a1,a0
@     .word 0x5ae7952b
@     ret