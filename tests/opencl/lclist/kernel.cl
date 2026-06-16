#include "common.h"

// Concurrent sorted singly-linked list with hand-over-hand (lock-coupling)
// insertion -- a classic building block of concurrent ordered indexes / GPU
// databases. Each work-item walks the list from the head holding the
// predecessor's lock while it acquires the current node's lock, then splices
// its own node in once it finds the right position.
//
// This is a STRONG Independent-Thread-Scheduling motivator, distinct from the
// `lockht` baseline:
//
//   1. It deadlocks DETERMINISTICALLY on a SIMT reconvergence-stack (IPDOM)
//      device at the very first lock(HEAD): every warp lane contends for the
//      head lock, one wins and is parked at the spin loop's reconvergence
//      point, and the losers spin forever because the winner can never reach
//      the matching unlock (it is far past the reconvergence point).
//
//   2. It cannot be repaired by a recognizable-idiom `vx_pred` compiler pass.
//      The page-10 transform needs a single acquire -> SESE critical section
//      -> release that a thread completes and opts out of. Here a lane holds a
//      lock across an UNBOUNDED, data-dependent traversal that performs FURTHER
//      blocking acquisitions. There is no single critical section to match; a
//      lane is simultaneously a lock *holder* (excluding others) and a *waiter*
//      across many would-be rounds. Only true per-thread program counters
//      (ITS) provide the forward progress this needs.

inline void nl_acquire(volatile __global int* lk) {
  // test-and-set (amoswap): old==0 means acquired. cmpxchg would lower to LR/SC,
  // which does not yield a single winner across SIMT lanes.
  while (atomic_xchg(lk, 1) != 0)
    ;  // spin until acquired
}

inline void nl_release(volatile __global int* lk) {
  atomic_xchg(lk, 0);
}

__kernel void lclist(__global int* node_key,    // [N+2] keys (sentinels + data)
                     __global int* node_next,   // [N+2] chain links
                     __global int* node_lock) { // [N+2] per-node spin locks
  int gid = get_global_id(0);
  int mynode = gid + FIRST_DATA_NODE;
  int key = node_key[mynode];

  // Lock-coupling traversal: hold pred, acquire curr, advance, release pred.
  int pred = HEAD_NODE;
  nl_acquire(&node_lock[pred]);
  int curr = node_next[pred];
  nl_acquire(&node_lock[curr]);

  while (node_key[curr] < key) {
    nl_release(&node_lock[pred]);
    pred = curr;
    curr = node_next[curr];
    nl_acquire(&node_lock[curr]);
  }

  // Holding both pred and curr: splice mynode between them.
  node_next[mynode] = curr;
  node_next[pred] = mynode;

  nl_release(&node_lock[curr]);
  nl_release(&node_lock[pred]);
}
