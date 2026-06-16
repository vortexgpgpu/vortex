#ifndef COMMON_H
#define COMMON_H

// Number of hash buckets. Kept modest relative to the number of concurrent
// work-items so that lanes of the same warp routinely contend for the same
// bucket lock -- which is precisely the case the SIMT reconvergence stack
// cannot make forward progress on.
#define NUM_BUCKETS 256

#endif // COMMON_H
