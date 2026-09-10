#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include "common.h"

// vx_barrier.h documents gbarrier::arrive() as returning "phase (current
// generation number)" and gbarrier::wait(phase) as blocking "until generation
// > phase". Both statements only hold if a completed global-barrier
// generation actually ADVANCES the phase that arrive() reports.
//
// Every warp of every core arrives on the probe barrier, so its generation
// completes; the cores then rendezvous on a SECOND global barrier (a
// different id, hence a different gbar row and a different per-core slot) so
// that the probe generation is known to have finished everywhere; then the
// probe phase is sampled again.
//
// Sampling is done with arrive() rather than wait() on purpose: wait() would
// BLOCK forever if the phase never advances, turning a data mismatch into a
// timeout with no diagnostics. arrive() is non-blocking, so the test always
// terminates and always reports the two observed phases.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  vortex::gbarrier probe(GBAR_PROBE_ID);
  vortex::gbarrier sync(GBAR_SYNC_ID);

  auto p0_ptr = reinterpret_cast<uint32_t*>(arg->p0_addr);
  auto p1_ptr = reinterpret_cast<uint32_t*>(arg->p1_addr);

  auto cid = vx_core_id();
  auto wid = vx_warp_id();
  auto tid = vx_thread_id();

  // Generation N of the probe barrier: every active warp of every core
  // arrives, so every core forwards and the cluster releases.
  uint32_t p0 = probe.arrive();

  // Rendezvous on the other global barrier. Its own round trip through the
  // cluster unit is issued after the probe's, so once this returns the probe
  // generation has completed on every core.
  sync.arrive_and_wait();

  // Sampled at the arrival of generation N+1, i.e. after generation N
  // completed: the documented phase must have moved.
  uint32_t p1 = probe.arrive();

  if (wid == 0 && tid == 0) {
    p0_ptr[cid] = p0;
    p1_ptr[cid] = p1;
  }
}
