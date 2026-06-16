#include "common.h"

// Range / interval locking (a real concurrent-database pattern): each work-item
// locks a WINDOW-wide window of consecutive locks, all held SIMULTANEOUSLY, then
// updates the cells it has reserved. Neighbouring work-items use overlapping
// windows, so many lanes hold many distinct, partially-overlapping lock sets at
// once.
//
// Locks are always acquired in ascending global index order, so the program is
// deadlock-free on a true MIMD machine.
//
// Purpose: validate the SCS "bounded-K limitation" case. Unlike lockht (1 lock)
// and lclist (2 locks at a time), here a lane holds WINDOW locks at once and
// overlapping windows keep WINDOW-deep holder contexts coexisting. With a SCS
// split table of depth K < WINDOW, forward progress requires the memory-backed
// split spill; without spill the table overflows. (On the current IPDOM stack
// it simply deadlocks at the first contended lock.)

inline void rl_lock(volatile __global int* l) {
  // test-and-set (amoswap): old==0 means acquired. cmpxchg would lower to LR/SC,
  // which does not yield a single winner across SIMT lanes.
  while (atomic_xchg(l, 1) != 0)
    ;  // spin until acquired
}

inline void rl_unlock(volatile __global int* l) {
  atomic_xchg(l, 0);
}

__kernel void rangelock(__global int* locks,   // [nlocks] per-cell spin locks
                        __global int* cell,    // [nlocks] reserved-cell counters
                        int nlocks) {
  int gid = get_global_id(0);
  int base = gid;   // window = [base, base + WINDOW), strictly ascending

  // Acquire all WINDOW locks, ascending index order (global order => no cycle).
  for (int i = 0; i < WINDOW; ++i)
    rl_lock(&locks[base + i]);


  // Critical section: bump every cell this lane has exclusively reserved.
  for (int i = 0; i < WINDOW; ++i)
    cell[base + i] += 1;


  // Release.
  for (int i = 0; i < WINDOW; ++i)
    rl_unlock(&locks[base + i]);
}
