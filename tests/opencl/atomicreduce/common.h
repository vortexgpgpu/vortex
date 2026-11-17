#ifndef COMMON_H
#define COMMON_H

// Counter slots of the atomic_global and atomic_local kernels.
#define OP_ADD     0
#define OP_SUB     1
#define OP_INC     2
#define OP_DEC     3
#define OP_MIN     4
#define OP_MAX     5
#define OP_UMIN    6
#define OP_UMAX    7
#define OP_AND     8
#define OP_OR      9
#define OP_XOR     10
#define OP_XCHG    11
#define OP_CMPXCHG 12
#define OP_FXCHG   13
#define NUM_OPS    14

// Start value of the exchange counters: outside the data range, so a chain
// that never stores leaves it in place and fails the membership check.
#define XCHG_INIT  12345

// Per-work-item return values of the atomic_global kernel.
#define RET_INC    0
#define RET_XCHG   1
#define RET_FXCHG  2
#define NUM_RETS   3

#endif // COMMON_H
