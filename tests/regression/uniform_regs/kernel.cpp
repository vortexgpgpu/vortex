#include <vx_spawn2.h>
#include <vx_uniform.h>
#include <VX_types.h>
#include "common.h"

// u1 args, u2 n, u3 src, u4 dst, u5 scale, u12 results, u6.. scratch
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  vx_u_mvx(1, arg);
  vx_u_load(VX_U_W,  2, 1, ARG_N);
  vx_u_load(VX_U_W,  5, 1, ARG_SCALE);
  vx_u_load(VX_U_XL, 3, 1, ARG_SRC);
  vx_u_load(VX_U_XL, 4, 1, ARG_DST);
  vx_u_load(VX_U_XL, 12, 1, ARG_RES);

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t n = vx_x_mvu(2);
  if (idx < n) {
    // dst[i] = src[i] * scale + n, addressed from uniform bases
    size_t off = idx * 4;
    auto src = reinterpret_cast<const uint32_t*>(vx_xu_op(VX_U_ADD, off, 3));
    auto dst = reinterpret_cast<uint32_t*>(vx_xu_op(VX_U_ADD, off, 4));
    uint32_t r = vx_xu_op(VX_U_MUL, *src, 5);
    *dst = vx_xu_op(VX_U_ADD, r, 2);
  }

  vx_u_li(6, 1);
  vx_u_slli(6, 6, __riscv_xlen - 1);
  vx_u_srli(7, 6, __riscv_xlen - 3);
  vx_u_op(VX_U_MUL, 8, 2, 5);
  vx_u_op(VX_U_DIVU, 9, 8, 5);
  vx_u_lui(13, 0x80000);
  vx_u_srai(13, 13, __riscv_xlen - 1);
  vx_u_op(VX_U_SUB, 14, 2, 5);
  vx_u_csrr(16, VX_CSR_NUM_THREADS);

  // a uniform loop: u10 counts to n
  vx_u_li(10, 0);
  __asm__ volatile (
    "1: .insn i 0x12, 0, x10, x10, 1\n\t"
    ".insn b 0x62, 4, x10, x2, 1b"
    ::: "memory");

  vx_u_store(VX_U_W, 7, 12, RES_SHIFT * 4);
  vx_u_store(VX_U_W, 9, 12, RES_DIV * 4);
  vx_u_store(VX_U_W, 10, 12, RES_LOOP * 4);
  vx_u_store(VX_U_W, 13, 12, RES_LUI * 4);
  vx_u_store(VX_U_W, 14, 12, RES_SUB * 4);
  vx_u_store(VX_U_W, 16, 12, RES_NT * 4);

  size_t res = vx_x_mvu(12);
  vx_xu_store(VX_U_W, 2, res, RES_XSTORE * 4);

  size_t xn = vx_xu_load(VX_U_W, 1, ARG_N);
  vx_ux_store(VX_U_W, xn, 12, RES_XLOAD * 4);
  float fs = vx_xu_flw(1, ARG_SCALE);
  vx_ux_fsw(fs, 12, RES_FLOAD * 4);
  vx_ux_store(VX_U_W, vx_ux_op(VX_U_SUB, 2, size_t(1)), 12, RES_SUBF * 4);
  vx_ux_store(VX_U_W, vx_ux_op(VX_U_SLTU, 5, xn), 12, RES_SLTF * 4);

  // bltu u5 (scale), n: taken; bltu u2 (n), scale: not taken
  size_t xs = vx_x_mvu(5), br = 0;
  __asm__ volatile (
    ".insn b %3, 6, x5, %1, 1f\n\t"
    "ori %0, %0, 4\n"
    "1: .insn b %3, 6, x2, %2, 2f\n\t"
    "ori %0, %0, 1\n"
    "2:"
    : "+r"(br) : "r"(xn), "r"(xs), "i"(VX_UF_BRANCH));
  vx_ux_store(VX_U_W, br, 12, RES_BRF * 4);

  // warp control on uniform operands: tmc to the active mask, a barrier of
  // one warp, then arrive (the phase in u20) and wait on the warp's own id
  vx_u_csrr(17, VX_CSR_ACTIVE_THREADS);
  vx_u_csrr(18, VX_CSR_WARP_ID);
  vx_u_li(19, 1);
  __asm__ volatile (
    ".insn r %0, 0, 0, x0, x17, x0\n\t"
    ".insn r %0, 4, 0, x0, x18, x19\n\t"
    ".insn r %0, 6, 0, x20, x18, x19\n\t"
    ".insn r %0, 6, 0, x0, x18, x20"
    :: "i"(RISCV_CUSTOM0 & ~1) : "memory");
  vx_u_store(VX_U_W, 19, 12, RES_WCTL * 4);
}
