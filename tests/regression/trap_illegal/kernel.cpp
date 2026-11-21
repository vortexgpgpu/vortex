#include <vx_spawn2.h>
#include "common.h"

// Runs insn under a local trap handler, after the setup instructions pre. The
// handler records mcause and mepc relative to insn, then returns past it. FS
// goes back to Initial and mtvec to its old value afterwards.
#define TRAP_PROBE(pre, insn, cause, epc_offset)                           \
  __asm__ volatile(                                                     \
    ".option push\n\t"                                                  \
    ".option norvc\n\t"                                                 \
    "li %0, -1\n\t"                                                     \
    "li %1, -1\n\t"                                                     \
    "la t0, 1f\n\t"                                                     \
    "csrrw t1, mtvec, t0\n\t"                                           \
    pre                                                                 \
    "2:\n\t"                                                            \
    insn "\n\t"                                                         \
    "j 3f\n\t"                                                          \
    ".align 2\n"                                                        \
    "1:\n\t"                                                            \
    "csrr %0, mcause\n\t"                                               \
    "csrr t0, mepc\n\t"                                                 \
    "la t2, 2b\n\t"                                                     \
    "sub %1, t0, t2\n\t"                                                \
    "addi t0, t0, 4\n\t"                                                \
    "csrw mepc, t0\n\t"                                                 \
    "mret\n"                                                            \
    "3:\n\t"                                                            \
    "csrw mtvec, t1\n\t"                                                \
    "li t2, 0x2000\n\t"                                                 \
    "csrs mstatus, t2\n\t"                                              \
    ".option pop"                                                       \
    : "=&r"(cause), "=&r"(epc_offset)                                   \
    :                                                                   \
    : "t0", "t1", "t2", "ft0", "memory")

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto out = reinterpret_cast<trap_out_t*>(arg->dst_addr);

  long cause[NUM_PROBES], epc_offset[NUM_PROBES];
#define FS_OFF "li t2, 0x6000\n\tcsrc mstatus, t2\n"
  TRAP_PROBE(FS_OFF, "fadd.s ft0, ft0, ft0", cause[PROBE_FS_OFF_FADD], epc_offset[PROBE_FS_OFF_FADD]);
  TRAP_PROBE(FS_OFF, "flw ft0, 0(sp)", cause[PROBE_FS_OFF_FLW], epc_offset[PROBE_FS_OFF_FLW]);
  TRAP_PROBE(FS_OFF, "csrr t0, fflags", cause[PROBE_FS_OFF_FFLAGS], epc_offset[PROBE_FS_OFF_FFLAGS]);
  TRAP_PROBE(FS_OFF, "fmv.x.w t0, ft0", cause[PROBE_FS_OFF_FMV], epc_offset[PROBE_FS_OFF_FMV]);
  // beq-form x0, x0, +8 with funct3 2 and 3
  TRAP_PROBE("", ".word 0x00002463", cause[PROBE_BRANCH_FUNCT3_2], epc_offset[PROBE_BRANCH_FUNCT3_2]);
  TRAP_PROBE("", ".word 0x00003463", cause[PROBE_BRANCH_FUNCT3_3], epc_offset[PROBE_BRANCH_FUNCT3_3]);
  // SYSTEM, funct3 0, imm12 0x7b2 (DRET, a debug-mode instruction)
  TRAP_PROBE("", ".word 0x7b200073", cause[PROBE_SYSTEM_RESERVED], epc_offset[PROBE_SYSTEM_RESERVED]);
  // SYSTEM, funct3 4 (HLV.B, a hypervisor load)
  TRAP_PROBE("", ".word 0x60004073", cause[PROBE_SYSTEM_FUNCT3_4], epc_offset[PROBE_SYSTEM_FUNCT3_4]);
  TRAP_PROBE("", "wfi", cause[PROBE_WFI], epc_offset[PROBE_WFI]);
  for (int i = 0; i < NUM_PROBES; ++i) {
    out->probes[i].cause = cause[i];
    out->probes[i].epc_offset = epc_offset[i];
  }

  long ms_initial, ms_dirty;
  __asm__ volatile(
    "li t0, 0x6000\n\t"
    "csrc mstatus, t0\n\t"
    "li t0, 0x2000\n\t"
    "csrs mstatus, t0\n\t"
    "csrr %0, mstatus\n\t"
    "fadd.s ft0, ft0, ft0\n\t"
    "csrr %1, mstatus"
    : "=&r"(ms_initial), "=&r"(ms_dirty)
    :
    : "t0", "ft0");
  out->mstatus_initial = (unsigned long)ms_initial;
  out->mstatus_dirty   = (unsigned long)ms_dirty;
}
