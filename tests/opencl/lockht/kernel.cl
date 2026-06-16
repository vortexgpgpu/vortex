#include "common.h"

// Concurrent chained hash table (multimap) insert using per-bucket spin locks.
//
// This is a deliberately useful, idiomatic concurrent data structure: every
// work-item splices its own node onto the head of its bucket's chain while
// holding that bucket's lock. On a CPU, or on a Volta+ GPU with Independent
// Thread Scheduling, it runs correctly.
//
// On a SIMT reconvergence-stack (IPDOM) GPU -- which is Vortex today -- it
// DEADLOCKS whenever two lanes of the same warp target the same bucket:
//
//   while (atomic_cmpxchg(lk, 0, 1) != 0) ;   // spin-acquire
//
// One lane wins the CAS and exits the loop; it is then masked off and parked
// at the loop's reconvergence (post-dominator) point, waiting for the other
// lanes to also exit before any of them proceeds. The losing lanes keep
// spinning because the lock is held -- but the winner can never reach
// unlock(), which lives past the reconvergence point. No lane makes progress.

inline void bucket_lock_acquire(volatile __global int* lk) {
  // test-and-set (amoswap): old==0 means we acquired. cmpxchg would lower to
  // LR/SC, which does not yield a single winner across SIMT lanes.
  while (atomic_xchg(lk, 1) != 0)
    ;  // spin until the lock is free
}

inline void bucket_lock_release(volatile __global int* lk) {
  atomic_xchg(lk, 0);
}

inline uint hash(uint key) {
  return (key * 2654435761u) % NUM_BUCKETS;  // Knuth multiplicative hash
}

__kernel void lockht(__global const int* keys,        // input key per work-item
                     __global int* node_key,          // [N] node payload
                     __global int* node_next,         // [N] chain link
                     __global int* bucket_head,       // [NUM_BUCKETS] chain head (-1 = empty)
                     __global int* bucket_lock) {     // [NUM_BUCKETS] spin lock (0 = free)
  int gid = get_global_id(0);
  int key = keys[gid];
  uint b = hash((uint)key);

  node_key[gid] = key;

  bucket_lock_acquire(&bucket_lock[b]);

  // critical section: prepend this node to bucket b's chain
  node_next[gid] = bucket_head[b];
  bucket_head[b] = gid;

  bucket_lock_release(&bucket_lock[b]);
}
