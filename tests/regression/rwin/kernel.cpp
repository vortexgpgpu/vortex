#include <vx_spawn2.h>
#include "common.h"

// vx_rwin: custom-2, funct7 1; the window of each field is written into that
// field, rs3's into funct3. Every asm block leaves the windows at zero.
#define RWIN(d, s1, s2, s3) ".insn r 0x5B, " #s3 ", 1, x" #d ", x" #s1 ", x" #s2 "\n\t"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto src = reinterpret_cast<const uint32_t*>(arg->src_addr);
  auto dst = reinterpret_cast<uint32_t*>(arg->dst_addr);
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= arg->num_points)
    return;
  uint32_t a = src[gid];
  uint32_t b = a * 3 + 1;
  uint32_t* out = dst + gid * CHK_COUNT;
  uint32_t r0, r1, r2, r3, r4;

  // x37 = a, x69 = b, r0 = x37 + x69 (rs1 and rs2 in different windows)
  __asm__ volatile (
    RWIN(1, 0, 0, 0) "addi x5, %[a], 0\n\t"
    RWIN(2, 0, 0, 0) "addi x5, %[b], 0\n\t"
    RWIN(0, 1, 2, 0) "add %[r], x5, x5\n\t"
    RWIN(0, 0, 0, 0)
    : [r] "=&r"(r0) : [a] "r"(a), [b] "r"(b));

  // x102 = a + 1, x103 = b * b, r1 = x102 + x103, r2 = x0 + x102
  __asm__ volatile (
    RWIN(3, 0, 0, 0)
    "addi x6, %[a], 1\n\t"
    "mul x7, %[b], %[b]\n\t"
    RWIN(0, 3, 3, 0)
    "add %[r1], x6, x7\n\t"
    "add %[r2], x0, x6\n\t"
    RWIN(0, 0, 0, 0)
    : [r1] "=&r"(r1), [r2] "=&r"(r2) : [a] "r"(a), [b] "r"(b));

#ifdef __riscv_zfinx
  // x225 = a, r3 = x225, through FP moves (Zfinx FP operands are x registers)
  __asm__ volatile (
    RWIN(7, 0, 0, 0) "fsgnj.s x1, %[a], %[a]\n\t"
    RWIN(0, 7, 7, 0) "fsgnj.s %[r], x1, x1\n\t"
    RWIN(0, 0, 0, 0)
    : [r] "=&r"(r3) : [a] "r"(a));
#else
  // f225 = a, r3 = f225
  __asm__ volatile (
    RWIN(7, 0, 0, 0) "fmv.w.x f1, %[a]\n\t"
    RWIN(0, 7, 0, 0) "fmv.x.w %[r], f1\n\t"
    RWIN(0, 0, 0, 0)
    : [r] "=&r"(r3) : [a] "r"(a));
#endif

  // x200 = a ^ 0x55; normal code in between cannot reach it
  __asm__ volatile (RWIN(6, 0, 0, 0) "xori x8, %[a], 0x55\n\t" RWIN(0, 0, 0, 0) :: [a] "r"(a));
  uint32_t sum = 0;
  for (uint32_t i = 0; i < (a & 7) + 4; ++i) {
    sum += src[(gid + i) % arg->num_points] ^ i;
  }
  __asm__ volatile ("" :: "r"(sum));
  __asm__ volatile (RWIN(0, 6, 0, 0) "addi %[r], x8, 0\n\t" RWIN(0, 0, 0, 0) : [r] "=&r"(r4));

  out[CHK_FIELDS]   = r0;
  out[CHK_STICKY]   = r1;
  out[CHK_X0]       = r2;
  out[CHK_FP]       = r3;
  out[CHK_ISOLATED] = r4;
}
