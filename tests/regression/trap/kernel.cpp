#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Steps mepc past the trapping instruction and returns. Uses only t6, which
// the trap sites below declare clobbered.
asm(".section .text\n"
    ".align 4\n"
    ".global trap_handler\n"
    "trap_handler:\n"
    "  csrr t6, mepc\n"
    "  addi t6, t6, 4\n"
    "  csrw mepc, t6\n"
    "  mret\n");
extern "C" void trap_handler();

// Even warps trap with ECALL and odd warps with EBREAK, at different sites, so
// a return PC or cause shared between warps shows up as a mismatch.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto dst_ptr = reinterpret_cast<uint32_t*>(arg->dst_addr);
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uintptr_t handler = (uintptr_t)trap_handler;

  // mtvec is one per core: only the block's first warp installs the handler,
  // and every warp of the block, across the issue slices, must read it back.
  // A warp that does not see it reports the mismatch instead of trapping.
  if (threadIdx.x < (uint32_t)vx_num_threads()) {
    csr_write(VX_CSR_MTVEC, handler);
  }
  __syncthreads();
  uint32_t bad = (csr_read(VX_CSR_MTVEC) != handler) ? TRAP_BAD_TVEC : 0;

  bool odd = (vx_warp_id() & 1) != 0;
  for (uint32_t i = 0; bad == 0 && i < arg->iters; ++i) {
    uintptr_t site, epc;
    uint32_t cause;
    if (odd) {
      __asm__ volatile("la %0, 1f\n\tebreak\n1:\n\tcsrr %1, mepc\n\tcsrr %2, mcause"
                       : "=&r"(site), "=&r"(epc), "=&r"(cause) :: "t6", "memory");
      bad |= (cause != 3) ? TRAP_BAD_CAUSE : 0;
    } else {
      __asm__ volatile("la %0, 1f\n\tecall\n1:\n\tcsrr %1, mepc\n\tcsrr %2, mcause"
                       : "=&r"(site), "=&r"(epc), "=&r"(cause) :: "t6", "memory");
      bad |= (cause != 11) ? TRAP_BAD_CAUSE : 0;
    }
    bad |= (epc != site) ? TRAP_BAD_EPC : 0;
  }
  bad |= (csr_read(VX_CSR_MTVAL) != 0) ? TRAP_BAD_TVAL : 0;

  if (idx < arg->num_points) {
    dst_ptr[idx] = bad;
  }
}
