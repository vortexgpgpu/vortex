#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include "common.h"

// A count-1 barrier arrival completes its generation immediately, so it must
// advance that slot's phase: the phase returned by the NEXT arrival on the
// same slot must be the complement of the one this arrival returned. That is
// a per-slot invariant -- barrier slots are independent state.
//
// The invariant is probed while a SECOND warp arrives on a DIFFERENT slot in
// the same cycle window: warp 0 and warp 1 are released together by a gate
// barrier, so the scheduler issues their arrivals on consecutive cycles.
//
// Spacing: `pre` and `post` are separated by nops so the audit read is never
// itself the back-to-back partner of the arrival it is auditing.
//
// Self-correction: a detected miss leaves the slot one flip short, which
// would desynchronise the two probe slots and hide later occurrences (the
// two slots must hold EQUAL phases for the cross-slot interaction to be
// observable at all). One extra arrival restores the parity.
#define SPACER __asm__ volatile ("nop; nop; nop; nop; nop; nop; nop; nop" ::: "memory")

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto err_ptr  = reinterpret_cast<uint32_t*>(arg->err_addr);
  auto pre_ptr  = reinterpret_cast<uint32_t*>(arg->pre_addr);
  auto post_ptr = reinterpret_cast<uint32_t*>(arg->post_addr);

  auto cid = vx_core_id();
  auto wid = vx_warp_id();
  auto tid = vx_thread_id();

  vortex::barrier       gate(GATE_ID);          // every warp of this core
  vortex::group_barrier probe0(PROBE0_ID, 1);   // count-1 -> completes at once
  vortex::group_barrier probe1(PROBE1_ID, 1);

  // Exactly one warp per probe slot: two warps sharing a count-1 slot would
  // see each other's flips and the audit would be meaningless.
  vortex::group_barrier mine = (wid == 0) ? probe0 : probe1;

  uint32_t errors = 0, first_pre = 0, first_post = 0;

  for (uint32_t r = 0; r < arg->rounds; ++r) {
    // Released on the same cycle -> warp 0 and warp 1 present their arrivals
    // to the barrier unit on consecutive cycles, on different slots.
    gate.arrive_and_wait();

    if (wid < 2) {
      uint32_t pre = mine.arrive();
      SPACER;
      uint32_t post = mine.arrive();

      if (((pre ^ post) & 1) == 0) {
        if (errors == 0) { first_pre = pre; first_post = post; }
        ++errors;
        mine.arrive();  // restore the slot's parity for the next round
      }
    }
  }

  // One rendezvous before the writes so no warp races ahead into teardown.
  gate.arrive_and_wait();

  if (tid == 0) {
    uint32_t idx = cid * arg->num_warps + wid;
    err_ptr[idx]  = errors;
    pre_ptr[idx]  = first_pre;
    post_ptr[idx] = first_post;
  }
}
