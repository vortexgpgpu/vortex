#include <vx_spawn2.h>
#include "common.h"

// Each round points mtvec at its own handler and traps straight after the
// write, and the handler rewrites mepc right before returning: a trap entry
// that ran ahead of the mtvec write would land in the previous round's handler
// (or the reset vector), and an mret that ran ahead of the mepc write would
// return to the trapping instruction and trap again forever.
extern "C" void trap_handler_ecall();
extern "C" void trap_handler_ebreak();
asm (".p2align 2\n"
     "trap_handler_ecall:\n"
     "  csrr a1, mepc\n"
     "  csrr a2, mcause\n"
     "  ori  a2, a2, 0x100\n"
     "  addi t0, a1, 4\n"
     "  csrw mepc, t0\n"
     "  mret\n"
     ".p2align 2\n"
     "trap_handler_ebreak:\n"
     "  csrr a1, mepc\n"
     "  csrr a2, mcause\n"
     "  ori  a2, a2, 0x200\n"
     "  addi t0, a1, 4\n"
     "  csrw mepc, t0\n"
     "  mret\n");

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= arg->num_points)
    return;
  auto dst = (uint32_t*)arg->dst_addr + gid * (TRAP_WORDS * TRAP_ROUNDS);

  uintptr_t saved_mtvec;
  asm volatile ("csrr %0, mtvec" : "=r"(saved_mtvec));

  register uintptr_t site asm("a0");
  register uintptr_t epc asm("a1") = 0;
  register uintptr_t cause asm("a2") = 0;
  asm volatile ("csrw mtvec, %3\n"
                "la   a0, 1f\n"
                "1: ecall\n"
                : "=&r"(site), "+r"(epc), "+r"(cause)
                : "r"(&trap_handler_ecall)
                : "t0", "memory");
  dst[0] = site;
  dst[1] = epc;
  dst[2] = cause;

  epc = 0;
  cause = 0;
  asm volatile ("csrw mtvec, %3\n"
                "la   a0, 1f\n"
                "1: ebreak\n"
                : "=&r"(site), "+r"(epc), "+r"(cause)
                : "r"(&trap_handler_ebreak)
                : "t0", "memory");
  dst[3] = site;
  dst[4] = epc;
  dst[5] = cause;

  asm volatile ("csrw mtvec, %0" :: "r"(saved_mtvec));
}
