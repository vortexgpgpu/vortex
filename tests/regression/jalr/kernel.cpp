#include <vx_intrinsics.h>
#include "common.h"

int main() {
  kernel_arg_t* __UNIFORM__ arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

  if (vx_warp_id() != 0 || vx_thread_id() != 0)
    return 0;

  uint32_t value = 0;
  asm volatile(
    // x1 is prepared so that the misaligned decode path can immediately
    // execute a valid JALR to label 3 and produce a deterministic failure
    // value instead of aborting in the decoder.
    "la x1, 3f\n"
    // 1744 is chosen to match the immediate embedded in the overlapped
    // misaligned instruction below: jalr x0, -1744(x1).
    "addi x1, x1, 1744\n"
    "la t0, 1f\n"
    // Add 1 on purpose. A correct JALR implementation must clear bit 0 and
    // land at 1f. An incorrect implementation enters at 1f + 1 instead.
    "addi t0, t0, 1\n"
    "jalr x0, t0, 0\n"
    ".balign 4\n"
    "1:\n"
    // 0x00806713 is a hand-picked overlapped instruction word:
    // - from the aligned entry at 1f it decodes as: ori x14, x0, 8
    // - from the misaligned entry at 1f + 1 it decodes as:
    //   jalr x0, -1744(x1)
    //
    // That lets both fixed and unfixed builds decode valid instructions while
    // still producing different results.
    ".4byte 0x00806713\n"
    // This is the next aligned instruction after the pass-path overlap word.
    ".4byte 0x00000093\n"
    "j 4f\n"
    "3:\n"
    // Fail signature used when JALR does not clear bit 0.
    "li %[value], 0x2468ace0\n"
    "j 5f\n"
    "4:\n"
    // Pass signature used when JALR correctly clears bit 0.
    "li %[value], 0x13579bdf\n"
    "5:\n"
    : [value] "=r" (value)
    :
    : "x1", "t0", "x14");

  uint32_t* dst_ptr = (uint32_t*)arg->dst_addr;
  dst_ptr[vx_core_id()] = value;
  return 0;
}
