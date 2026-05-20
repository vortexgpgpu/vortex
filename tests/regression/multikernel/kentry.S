// Multi-entry .vxbin test — per-kernel KMU entry stubs.
//
// Hand-written equivalent of what POCL's compile_vortex_program emits:
// one naked stub per kernel, placed in the .vx_entry section (KEEP'd by
// the kernel link script so --gc-sections cannot drop it). Each stub
// loads its kernel's address into s11 — callee-saved, so it survives
// the prologue's __init_tls / __libc_init_array calls — and jumps to
// the shared __vx_cta_entry prologue in libvortex2.a. vxbin.py records
// each __vx_kentry_<name> symbol in the VXSYMTAB footer; the runtime's
// vx_module_get_kernel("<name>") resolves to the stub's address.

.section .vx_entry,"ax",@progbits

.global __vx_kentry_add_k
.type   __vx_kentry_add_k, @function
__vx_kentry_add_k:
  lla  s11, add_k
  j    __vx_cta_entry
.size   __vx_kentry_add_k, .-__vx_kentry_add_k

.global __vx_kentry_mul_k
.type   __vx_kentry_mul_k, @function
__vx_kentry_mul_k:
  lla  s11, mul_k
  j    __vx_cta_entry
.size   __vx_kentry_mul_k, .-__vx_kentry_mul_k

.global __vx_kentry_acc_k
.type   __vx_kentry_acc_k, @function
__vx_kentry_acc_k:
  lla  s11, acc_k
  j    __vx_cta_entry
.size   __vx_kentry_acc_k, .-__vx_kentry_acc_k
