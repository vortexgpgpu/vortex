#include <vx_spawn2.h>
#include <vx_uniform.h>
#include <VX_types.h>
#include "common.h"

// u1 args, u2 n, u3 src, u4 dst, u5 scale, u6.. scratch
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  vx_u_mvx(1, arg);
  vx_u_load(VX_U_WU, 2, 1, ARG_N);
  vx_u_load(VX_U_WU, 5, 1, ARG_SCALE);
  vx_u_load(VX_U_D,  3, 1, ARG_SRC);
  vx_u_load(VX_U_D,  4, 1, ARG_DST);
  vx_u_load(VX_U_D, 12, 1, ARG_RES);

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t n = vx_x_mvu(2);
  if (idx < n) {
    // dst[i] = src[i] * scale + n, through 64-bit uniform bases
    uint32_t off = idx * 4;
    uint32_t v = vx_u_gload(VX_U_W, 3, off);
    uint32_t r = vx_xu_op(VX_U_MUL, v, 5);
    r = vx_xu_op(VX_U_ADD, r, 2);
    vx_u_gstore(VX_U_W, 4, off, r);
  }

  // 64-bit uniform arithmetic at either XLEN
  vx_u_li(6, 1);
  vx_u_slli(6, 6, 40);
  vx_u_srli(7, 6, 38);
  vx_u_op(VX_U_MUL, 8, 2, 5);
  vx_u_op(VX_U_DIVU, 9, 8, 5);
  vx_u_lui(13, 0x80000);
  vx_u_srli(13, 13, 32);
  vx_u_li(14, -1);
  vx_u_srli(14, 14, 33);           // 0x7fffffff
  vx_u_opi(0, 15, 0, 1);
  vx_u_opw(VX_U_ADD, 15, 14, 15);  // 0x7fffffff + 1 -> 0xffffffff80000000
  vx_u_csrr(16, VX_CSR_NUM_THREADS);

  // a uniform loop: u10 counts to n
  vx_u_li(10, 0);
  __asm__ volatile (
    "1: .insn i 0x12, 0, x10, x10, 1\n\t"
    ".insn b 0x62, 4, x10, x2, 1b"
    ::: "memory");

  vx_u_store(VX_U_D, 6, 12, RES_SHL40 * 4);
  vx_u_store(VX_U_W, 7, 12, RES_SRL * 4);
  vx_u_store(VX_U_W, 9, 12, RES_DIV * 4);
  vx_u_store(VX_U_W, 10, 12, RES_LOOP * 4);
  vx_u_store(VX_U_W, 13, 12, RES_LUI_HI * 4);
  vx_u_srli(15, 15, 32);
  vx_u_store(VX_U_W, 15, 12, RES_ADDW * 4);
  vx_u_store(VX_U_W, 16, 12, RES_NT * 4);

  size_t res = vx_x_mvu(12);
  vx_xu_store(VX_U_W, 2, res, RES_XSTORE * 4);
}
