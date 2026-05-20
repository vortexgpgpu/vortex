// Multi-entry .vxbin test — device side, C-runtime-prologue edition.
//
// Three independent kernels in one program, each its own KMU entry point
// (see kentry.S). Unlike a bare add/mul pair, these kernels deliberately
// depend on state that exists ONLY once __vx_cta_entry's prologue has run:
//
//   * g_cubes[] is filled by build_cube_table(), registered via
//     __attribute__((constructor)) into .init_array. __vx_cta_entry calls
//     __libc_init_array, which runs it. Skip that call and g_cubes[] stays
//     zero-initialized (.bss) — every result comes out wrong, not crashed.
//
//   * t_tag (initialized => .tdata) and t_acc (zero-init => .tbss) are
//     thread_local. __vx_cta_entry points tp at this hart's own TLS image
//     and calls __init_tls, which copies the .tdata template in and zeroes
//     .tbss. Having both makes the image span .tdata + .tbss, so the
//     per-hart tp stride in vx_start.S must be __tbss_offset + __tbss_size
//     — a too-small stride overlaps adjacent harts' images. acc_k reaches
//     both through noinline helpers, so the accesses are genuine
//     tp-relative loads/stores.
//
// The multi-entry mechanism (VXSYMTAB footer, per-name resolution, the
// shared __vx_cta_entry prologue, vx_enqueue_launch's per-kernel PC) is
// still exercised — three named kernels resolved out of one .vxbin.
//
// __vx_cta_entry loads a0 with VX_CSR_MSCRATCH before calling the kernel,
// so each kernel takes the kernel_arg_t pointer as its first argument.
// This program defines no kernel_main: it is a footer-bearing multi-entry
// .vxbin, so vx_start.S's dead _start resolves the weak kernel_main to 0.

#include <vx_spawn2.h>
#include "common.h"

// --- exercises __libc_init_array -------------------------------------------
// External linkage + .bss: the compiler cannot prove the post-constructor
// contents, so kernel reads stay real loads and the constructor stays an
// .init_array entry (never folded into static data).
int32_t g_cubes[16];

__attribute__((constructor))
static void build_cube_table() {
  for (int i = 0; i < 16; ++i)
    g_cubes[i] = (i + 1) * (i + 1) * (i + 1);   // 1, 8, 27, ..., 4096
}

// --- exercises __init_tls --------------------------------------------------
// t_tag is INITIALIZED, so it lands in .tdata: __init_tls copies that
// template into every hart's TLS image. t_acc is zero-init (.tbss). Having
// both makes each hart's image span .tdata + .tbss (__tbss_offset > 0) —
// the case the per-hart tp stride in vx_start.S must size correctly.
thread_local int32_t t_tag = 0x5A5A;
thread_local int64_t t_acc;

// noinline so the thread-local accesses are emitted as real tp-relative
// loads/stores rather than hoisted into plain locals.
__attribute__((noinline)) static void    tls_add(int32_t x) { t_acc += x; }
__attribute__((noinline)) static int32_t tls_tag()          { return t_tag; }

static inline uint32_t global_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

// add_k: dst[i] = src[i] + cubes[i % 16]
__kernel void add_k(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t i = global_id();
  if (i >= arg->count)
    return;
  auto src = reinterpret_cast<int32_t*>(arg->src_addr);
  auto dst = reinterpret_cast<int32_t*>(arg->dst_addr);
  dst[i] = src[i] + g_cubes[i & 15];
}

// mul_k: dst[i] = src[i] * cubes[i % 16]
__kernel void mul_k(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t i = global_id();
  if (i >= arg->count)
    return;
  auto src = reinterpret_cast<int32_t*>(arg->src_addr);
  auto dst = reinterpret_cast<int32_t*>(arg->dst_addr);
  dst[i] = src[i] * g_cubes[i & 15];
}

// acc_k: dst[i] = src[i] + sum_{j=0..7} cubes[(i + j) % 16] + t_tag,
// summing through the .tbss accumulator and folding in the .tdata tag.
// Correctness needs each hart's TLS image to be private and non-
// overlapping: a too-small per-hart stride lets one hart's t_acc writes
// stomp a neighbour's t_tag.
__kernel void acc_k(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t i = global_id();
  if (i >= arg->count)
    return;
  auto src = reinterpret_cast<int32_t*>(arg->src_addr);
  auto dst = reinterpret_cast<int32_t*>(arg->dst_addr);
  t_acc = 0;                                  // this hart's own .tbss copy
  for (uint32_t j = 0; j < 8; ++j)
    tls_add(g_cubes[(i + j) & 15]);
  dst[i] = src[i] + static_cast<int32_t>(t_acc) + tls_tag();
}
