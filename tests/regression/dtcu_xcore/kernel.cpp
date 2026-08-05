#include "common.h"
#include <vx_spawn2.h>
#include <vx_dtensor.h>
#include <vx_intrinsics.h>

// dtcu_xcore: the only test where the core that OBSERVES completion is not the core
// that SUBMITTED the descriptor. That separation is the entire reason the completion
// flag lives in the descriptor rather than in engine state -- a ticket returned in a
// register never reaches another core -- and until this test existed nothing exercised
// it. dtcu_basic, dtcu_compare and cgo27_motivation all submit and check from the same
// warp of the same core.
//
// Roles are assigned by ATOMIC TICKET, not by block index and not by core id:
//   * block -> core is not a contract. The scheduler pulls at most one CTA per core per
//     cycle from a shared queue, which happens to spread one CTA per core when they all
//     start idle, but nothing guarantees it.
//   * keying the submitter off vx_core_id() == 0 would deadlock every consumer in any
//     dispatch where no CTA landed on core 0.
// The dispenser always hands out a ticket 0, so a submitter always exists and the
// consumers' wait always terminates. WHICH cores the tickets landed on is recorded here
// and asserted by the host: a run where every CTA landed on one core is a FAILED test,
// not a silently vacuous pass.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  if (vx_thread_id() != 0)
    return; // one thread per CTA does the whole job

  xcore_ctl_t* ctl = (xcore_ctl_t*)(uintptr_t)arg->ctl_addr;
  const uint32_t cid = (uint32_t)vx_core_id();

  // amoadd resolves at the LLC, so tickets are globally unique across cores.
  const uint32_t t = __atomic_fetch_add(&ctl->arrive, 1u, __ATOMIC_ACQ_REL);
  if (t >= XCORE_MAX_CTAS)
    return;
  __atomic_fetch_or(&ctl->core_of_ticket[t], cid + 1u, __ATOMIC_RELEASE);

  if (t == 0) {
    // ---- SUBMITTER ----
    if (arg->engine == DTCU_ENGINE_CLUSTER) {
      while (0 == dtensor_cluster_start(arg->desc_addr))
        ;
    } else {
      while (0 == dtensor_socket_start(arg->desc_addr))
        ;
    }
    // Publish "in flight" only AFTER the start retires. This is what makes the consumers
    // begin spinning while done is still 0, so they observe the 0 -> 1 transition rather
    // than walking up to a flag that was already set -- which would prove nothing.
    __atomic_fetch_or(&ctl->armed, 1u, __ATOMIC_RELEASE);

    // Safety net, NOT the evidence. The submitter also waits so D is complete before the
    // kernel retires even in a degenerate dispatch where every CTA landed here and no
    // consumer exists. The cross-core property is proved by done_seen/core_of_ticket,
    // which this branch never touches.
    for (uint32_t i = 0; i < XCORE_SPIN_LIMIT; ++i)
      if (dtensor_check(arg->desc_addr)) break;
    return;
  }

  // ---- CONSUMER: never issued the descriptor, may not even be in the same socket ----
  uint32_t i;
  for (i = 0; i < XCORE_SPIN_LIMIT; ++i)
    if (__atomic_fetch_or(&ctl->armed, 0u, __ATOMIC_ACQUIRE)) break;
  if (i == XCORE_SPIN_LIMIT) {
    __atomic_fetch_or(&ctl->done_seen[t], 2u, __ATOMIC_RELEASE);
    return;
  }

  for (i = 0; i < XCORE_SPIN_LIMIT; ++i)
    if (dtensor_check(arg->desc_addr)) break;
  if (i == XCORE_SPIN_LIMIT) {
    __atomic_fetch_or(&ctl->done_seen[t], 2u, __ATOMIC_RELEASE);
    return;
  }

  // done == 1 has to imply D is visible TO THIS CORE, which the flag alone does not
  // cover. Prove it with ordinary loads from a core that has never touched D: they miss
  // this core's L1 and must resolve to what the engine actually wrote. The LAST element
  // is read as well as the first, because "the flag beat the final store" is exactly the
  // ordering bug the ack-based store completion exists to prevent.
  const uint32_t* pD = (const uint32_t*)(uintptr_t)arg->D_addr;
  const uint32_t d0 = pD[0];
  const uint32_t dl = pD[arg->M * arg->N - 1];
  __atomic_fetch_or(&ctl->witness[t], xcore_witness_of(d0, dl), __ATOMIC_RELEASE);
  __atomic_fetch_or(&ctl->done_seen[t], 1u, __ATOMIC_RELEASE);
  __atomic_fetch_add(&ctl->observers, 1u, __ATOMIC_ACQ_REL);
}
