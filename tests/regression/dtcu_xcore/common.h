#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <dtcu_cfg.h>   // DTCU_ENGINE_{SOCKET,CLUSTER}

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// Upper bound on CTAs; the host launches exactly VX_CFG_NUM_CORES of them.
#define XCORE_MAX_CTAS 64

// Spin bound. The failure this test exists to catch is "the completion flag never
// becomes visible to another core", and unbounded that is not a test failure but an
// infinite simulator run.
//
// The bound has to be small enough to FAIL FAST, which is a stronger constraint than it
// looks: every iteration is an AMO round trip to the LLC, and a simulator executes those
// at a few thousand per wall-clock second. Measured with the flag deliberately polled by
// a plain load (so it can never become visible), a 20,000,000 bound did not terminate in
// ten minutes -- indistinguishable from a hang, which defeats the purpose of having a
// bound at all.
//
// 200,000 iterations is the compromise. The whole cluster-engine case above completes in
// ~45,000 simulated cycles and one AMO round trip is at least ~10 cycles, so this covers
// upwards of 40x the entire run: it cannot false-positive on a slow-but-working engine,
// and it reports a real hang in seconds.
#define XCORE_SPIN_LIMIT 200000u

// Shared control block, one device buffer. Every field is touched with an AMO because
// producer and consumer are different cores: a plain load installs the line in the
// reader's L1 and every later read returns that stale copy.
typedef struct {
  uint32_t arrive;     // AMO ticket dispenser; ticket 0 is the submitter
  uint32_t armed;      // set by the submitter AFTER its start instruction retires
  uint32_t observers;  // consumers that saw done
  uint32_t pad;
  uint32_t core_of_ticket[XCORE_MAX_CTAS]; // 1 + vx_core_id(); 0 means never ran
  uint32_t done_seen[XCORE_MAX_CTAS];      // bit 0 = saw done, bit 1 = hit the spin limit
  uint32_t witness[XCORE_MAX_CTAS];        // D as the CONSUMER read it, hashed
} xcore_ctl_t;

typedef struct {
  uint32_t engine;     // DTCU_ENGINE_{SOCKET,CLUSTER}
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t D_addr;
  uint64_t desc_addr;
  uint64_t ctl_addr;
} kernel_arg_t;

// One hash, defined once, computed identically on device and host. The consumer folds
// the first and last D words through it; the host recomputes it from its own read-back
// and compares. Two separate expressions here would be a bug waiting to happen.
static inline uint32_t xcore_witness_of(uint32_t d_first, uint32_t d_last) {
  return d_first ^ (d_last * 2654435761u);
}

#endif
