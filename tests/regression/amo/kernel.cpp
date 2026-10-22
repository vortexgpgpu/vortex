#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Retry bound for the local-memory LR/SC loop. High enough that genuine
// contention between the harts of a group always resolves, low enough that a
// store-conditional which never succeeds fails the case instead of hanging it.
static const uint32_t LRSC_MAX_RETRIES = 4096;

// All harts contend on a single shared word. Each test exercises a
// different RVA primitive; the host validates the closed-form expected
// final value against the post-run shared word, plus per-hart "observed
// old value" / "iterations completed" counters where applicable.

typedef void (*PFN_Kernel)(kernel_arg_t* __UNIFORM__ arg);

// Globally-unique hart id (mhartid): (cid * NUM_WARPS + wid) * NUM_THREADS
// + tid, matching the LLC reservation table's make_hart_id.
static inline uint32_t hart_id() {
  return (uint32_t)vx_hart_id();
}

// 1) AMOADD hammer.
//   each hart does N amoadd.w +1 → final = num_harts * iters.
void kernel_amoadd(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (int32_t*)arg->shared_addr;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    __atomic_fetch_add(shared, 1, __ATOMIC_SEQ_CST);
  }
}

// 2) AMOOR hammer.
//   each hart sets its own bit (mod 32). After all harts, every used
//   bit is set; verify final == ((1 << min(num_harts,32)) - 1) for ≤32
//   harts. With more harts, multiple harts share bits; final is just
//   any value with all those bits set (they're idempotent).
void kernel_amoor(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (uint32_t*)arg->shared_addr;
  uint32_t bit = 1u << (hart_id() & 31);
  for (uint32_t i = 0; i < arg->iters; ++i) {
    __atomic_fetch_or(shared, bit, __ATOMIC_SEQ_CST);
  }
}

// 3) AMOAND hammer.
//   start with all bits set; each hart clears its own bit (mod 32).
//   For num_harts ≥ 32, final = 0; for fewer, final = ~mask of cleared bits.
void kernel_amoand(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (uint32_t*)arg->shared_addr;
  uint32_t bit = ~(1u << (hart_id() & 31));
  for (uint32_t i = 0; i < arg->iters; ++i) {
    __atomic_fetch_and(shared, bit, __ATOMIC_SEQ_CST);
  }
}

// 4) AMOXOR hammer.
//   each hart XORs its bit twice (per iter). Net effect per iter = 0,
//   so final = initial value. Tests that XOR commit + return is sane
//   under contention even if final is invariant.
void kernel_amoxor(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (uint32_t*)arg->shared_addr;
  uint32_t bit = 1u << (hart_id() & 31);
  for (uint32_t i = 0; i < arg->iters; ++i) {
    __atomic_fetch_xor(shared, bit, __ATOMIC_SEQ_CST);
    __atomic_fetch_xor(shared, bit, __ATOMIC_SEQ_CST);
  }
}

// 5) AMOMAX hammer (signed).
//   each hart amomax.w with its hart_id. Final = num_harts - 1.
void kernel_amomax(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (int32_t*)arg->shared_addr;
  int32_t hid = (int32_t)hart_id();
  for (uint32_t i = 0; i < arg->iters; ++i) {
    int32_t expected = __atomic_load_n(shared, __ATOMIC_SEQ_CST);
    int32_t prev;
    asm volatile ("amomax.w %0, %2, (%1)"
                  : "=r"(prev)
                  : "r"(shared), "r"(hid)
                  : "memory");
    (void)expected; (void)prev;
  }
}

// 6) AMOMINU hammer (unsigned).
//   start with shared = 0xFFFFFFFF; each hart amominu.w with its hart_id.
//   Final = 0 (lowest hart_id wins).
void kernel_amominu(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (uint32_t*)arg->shared_addr;
  uint32_t hid = hart_id();
  for (uint32_t i = 0; i < arg->iters; ++i) {
    uint32_t prev;
    asm volatile ("amominu.w %0, %2, (%1)"
                  : "=r"(prev)
                  : "r"(shared), "r"(hid)
                  : "memory");
    (void)prev;
  }
}

// 7) AMOSWAP exchange.
//   each hart swaps in its hart_id repeatedly. The host accepts any
//   final value that is a valid hart_id. Per-hart buffer captures each
//   hart's last observed-old value to verify it's a previously-written
//   hart_id (not garbage).
void kernel_amoswap(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (uint32_t*)arg->shared_addr;
  auto observed = (uint32_t*)arg->per_hart_addr;
  uint32_t hid = hart_id();
  uint32_t last_old = hid;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    last_old = __atomic_exchange_n(shared, hid, __ATOMIC_SEQ_CST);
  }
  observed[hid] = last_old;
}

// 8) LR/SC counter (lock-free CAS-style increment).
//   each hart increments shared by 1 via LR/SC retry. Final =
//   num_harts * iters. Per-hart buffer captures retry count.
void kernel_lrsc_counter(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (int32_t*)arg->shared_addr;
  auto retries = (uint32_t*)arg->per_hart_addr;
  uint32_t hid = hart_id();
  uint32_t r = 0;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    int32_t old, sc_fail;
    do {
      asm volatile ("lr.w %0, (%1)"
                    : "=r"(old)
                    : "r"(shared)
                    : "memory");
      int32_t newv = old + 1;
      asm volatile ("sc.w %0, %2, (%1)"
                    : "=r"(sc_fail)
                    : "r"(shared), "r"(newv)
                    : "memory");
      if (sc_fail) ++r;
    } while (sc_fail);
  }
  retries[hid] = r;
}

// 8) AMOADD.W.AQRL hammer.
//   Same closed-form as kernel_amoadd, but the AMO carries the
//   acquire+release ordering bits. SimX is sequentially consistent,
//   so the result must be identical to the plain variant — this test
//   verifies the .aqrl encoding decodes correctly and that the bank
//   commits the RMW the same way it does for plain AMOADD.
//   Final = num_harts * iters.
void kernel_amoadd_aqrl(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (int32_t*)arg->shared_addr;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    int32_t one = 1;
    int32_t prev;
    asm volatile ("amoadd.w.aqrl %0, %2, (%1)"
                  : "=r"(prev)
                  : "r"(shared), "r"(one)
                  : "memory");
    (void)prev;
  }
}

// 9) LR.W.AQ / SC.W.RL counter (canonical acquire/release idiom).
//   Same lock-free CAS retry as kernel_lrsc_counter, but with the
//   acquire bit on LR and the release bit on SC. Functionally
//   equivalent under SimX's SC ordering; verifies the .aq/.rl
//   encoding paths through decode and the bank.
//   Final = num_harts * iters.
void kernel_lrsc_counter_aqrl(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (int32_t*)arg->shared_addr;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    int32_t old, sc_fail;
    do {
      asm volatile ("lr.w.aq %0, (%1)"
                    : "=r"(old)
                    : "r"(shared)
                    : "memory");
      int32_t newv = old + 1;
      asm volatile ("sc.w.rl %0, %2, (%1)"
                    : "=r"(sc_fail)
                    : "r"(shared), "r"(newv)
                    : "memory");
    } while (sc_fail);
  }
}

// 10) Atomic reduction.
//   Every hart reads one element from per_hart[] and atomicAdds it
//   into the single global accumulator at shared_addr. Final =
//   sum_{h=0..n-1} per_hart[h]. With per_hart[h] = h+1 (set by host),
//   final = n*(n+1)/2.
void kernel_atomic_reduction(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (int32_t*)arg->shared_addr;
  auto in     = (int32_t*)arg->per_hart_addr;
  uint32_t hid = hart_id();
  __atomic_fetch_add(shared, in[hid], __ATOMIC_SEQ_CST);
}

// 11) Atomic critical section (spinlock).
//   Uses amoswap.w to atomically swap 1 in and read back the prior value;
//   success when prior was 0. Release via amoswap.w with 0.
//
//   IMPORTANT — SIMD-lockstep gate: this pattern deadlocks within a
//   warp on architectures with strict warp-lockstep execution (Vortex,
//   pre-Volta CUDA): one thread acquires the lock and diverges into
//   the critical section, but the IPDOM stack stalls it until the
//   other threads in the warp leave the else-branch spin loop — which
//   they can't, because that requires the holder to release.
//   The workaround is to serialize at warp granularity: only one thread per warp
//   participates in the lock dance. Inter-warp contention still
//   exercises the AMO bank fully.
//   Final counter = num_active_harts * iters, where num_active_harts =
//   num_cores * num_warps (one thread per warp).
void kernel_atomic_critical(kernel_arg_t* __UNIFORM__ arg) {
  if (vx_thread_id() != 0) return;
  auto counter = (int32_t*)arg->shared_addr;
  auto lock    = (int32_t*)arg->per_hart_addr;  // [0] is the lock
  for (uint32_t i = 0; i < arg->iters; ++i) {
    // Acquire: amoswap.w with 1; succeeds when prior == 0.
    int32_t prev;
    do {
      asm volatile ("amoswap.w %0, %2, (%1)"
                    : "=r"(prev)
                    : "r"(lock), "r"(1)
                    : "memory");
    } while (prev != 0);

    // Critical section: non-atomic increment.
    int32_t v = *counter;
    *counter = v + 1;

    // Release: amoswap.w with 0 (atomicExch in CUDA).
    int32_t zero = 0, unused;
    asm volatile ("amoswap.w %0, %2, (%1)"
                  : "=r"(unused)
                  : "r"(lock), "r"(zero)
                  : "memory");
    (void)unused;
  }
}

// (LR/SC mutex omitted: a spin-lock acquire deadlocks under Vortex's
// SIMD-lockstep semantics. Same issue as kernel_atomic_critical but
// using LR/SC instead of amoswap; the warp-gated workaround used in
// kernel_atomic_critical is also the right way to write it.)

// 12) Issuer self-consistency.
//   Each hart operates on its OWN private word (no cross-hart sharing):
//   a plain load (which caches the line), an atomic +1, then a plain
//   load that MUST observe the increment — RISC-V same-hart program
//   order requires a hart to see its own prior atomic. The observed
//   value is recorded for the host. Fails when a non-LLC cache leaves
//   the issuer's cached copy un-invalidated after the AMO (the post-AMO
//   load then hits the stale line). Final per_hart[hid] == 1.
void kernel_self_consistency(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t hid = hart_id();
  // Each hart owns a private 64B cache line (no false sharing): shared + hid*64.
  volatile uint32_t* p = (volatile uint32_t*)((uint8_t*)arg->shared_addr + (size_t)hid * 64);
  uint32_t* per_hart = (uint32_t*)arg->per_hart_addr;
  (void)*p;  // plain load: bring this hart's line into the L1
  __atomic_fetch_add((uint32_t*)p, 1u, __ATOMIC_SEQ_CST);
  per_hart[hid] = *p;  // plain load after the AMO: must observe the +1
}

// 13) LMEM AMOADD hammer.
//   The same contention as test 0, moved onto local memory so the
//   read-modify-write is resolved by the LMEM banks instead of the dcache.
//   Local memory is per-workgroup and the host cannot read it, so each group's
//   leader publishes its group's final count into its own slot of the per-hart
//   buffer. Every hart contributes `iters` increments to exactly one group, so
//   the published values sum to num_harts * iters however the groups are
//   shaped -- a closed form that does not depend on the launch geometry.
void kernel_amoadd_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem     = (int32_t*)__local_mem();
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();

  if (threadIdx.x == 0) {
    *lmem = 0;
  }
  __syncthreads();

  for (uint32_t i = 0; i < arg->iters; ++i) {
    __atomic_fetch_add(lmem, 1, __ATOMIC_SEQ_CST);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    per_hart[hid] = (uint32_t)*lmem;
  }
}

// 14) LMEM LR/SC counter.
//   The lock-free retry of test 7, moved onto local memory so the reservation
//   is resolved by the LMEM banks. Each group's leader publishes its group's
//   final count, so the published values sum to num_harts * iters whatever the
//   launch geometry -- the same closed form test 13 uses.
//
//   The retry is bounded. A store-conditional that never reports success would
//   otherwise spin forever, and a test that hangs reports nothing; with the
//   bound, the published total comes up short and the case fails.
void kernel_lrsc_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem     = (int32_t*)__local_mem();
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  // Slot 1 (byte offset 4) sits in the upper lane of a 64-bit hart's lmem
  // bank word, so the contended loop exercises the sub-word LR/SC datapath
  // and the SC-outcome lane; on a 32-bit hart it is an ordinary word.
  auto counter  = &lmem[1];

  if (threadIdx.x == 0) {
    *counter = 0;
  }
  __syncthreads();

  for (uint32_t i = 0; i < arg->iters; ++i) {
    int32_t old, sc_fail;
    uint32_t guard = 0;
    do {
      asm volatile ("lr.w %0, (%1)"
                    : "=r"(old)
                    : "r"(counter)
                    : "memory");
      int32_t newv = old + 1;
      asm volatile ("sc.w %0, %2, (%1)"
                    : "=r"(sc_fail)
                    : "r"(counter), "r"(newv)
                    : "memory");
      ++guard;
    } while (sc_fail && guard < LRSC_MAX_RETRIES);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    per_hart[hid] = (uint32_t)*counter;
  }
}

#if VX_CFG_EXT_ZACAS_ENABLED
// 15) Native compare-and-swap contention.
//   Every hart races to claim each slot of a ladder with its own id, so for
//   slot i exactly one hart sees the empty value and swaps, and the others see
//   that winner's id. Each hart records how many slots it won.
//
//   The point is the per-hart return value, not just the final memory: a
//   compare-and-swap that returned the wrong old word would still leave the
//   ladder looking plausible while every loser believed it had won. The host
//   checks that the wins across harts sum to exactly the number of slots.
void kernel_cas_ladder(kernel_arg_t* __UNIFORM__ arg) {
  auto shared   = (uint32_t*)arg->shared_addr;
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  const uint32_t claim = hid + 1;   // 0 is the unclaimed marker
  uint32_t wins = 0;

  const uint32_t slots = (arg->iters < CAS_LADDER_SLOTS)
                       ? arg->iters : CAS_LADDER_SLOTS;
  for (uint32_t i = 0; i < slots; ++i) {
    uint32_t expected = 0;
    // Returns true only for the hart whose swap actually landed; `expected`
    // is updated with the observed word either way.
    if (__atomic_compare_exchange_n(&shared[i], &expected, claim, 0,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
      ++wins;
    }
  }
  per_hart[hid] = wins;
}

// 16) LMEM compare-and-swap ladder.
//   The cas_ladder race moved onto local memory, so the swap and the old word
//   it returns come from the LMEM banks. Each group races for its own ladder
//   and exactly one of its harts wins each slot, so the win counts sum to
//   groups * slots. The host cannot read local memory or the launch shape, so
//   the group count rides with each leader's publish as a marker well above
//   any possible win count -- the same closed-form idea as tests 13/14.
void kernel_cas_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem     = (uint32_t*)__local_mem();
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  const uint32_t claim = hid + 1;   // 0 is the unclaimed marker

  const uint32_t slots = (arg->iters < CAS_LMEM_SLOTS)
                       ? arg->iters : CAS_LMEM_SLOTS;
  if (threadIdx.x == 0) {
    for (uint32_t i = 0; i < slots; ++i) {
      lmem[i] = 0;
    }
  }
  __syncthreads();

  uint32_t wins = 0;
  for (uint32_t i = 0; i < slots; ++i) {
    uint32_t expected = 0;
    if (__atomic_compare_exchange_n(&lmem[i], &expected, claim, 0,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
      ++wins;
    }
  }
  per_hart[hid] = wins + ((threadIdx.x == 0) ? CAS_LMEM_GROUP_MARK : 0);
}

// 17) Compare-and-swap increment on global memory.
//   The classic lock-free counter: read, then swap in read+1 until no other hart
//   intervened. Unlike the ladders, every swap after the first compares against
//   a nonzero word the hart observed, and the counter starts from a nonzero
//   base, so a comparand that does not reach the memory side is caught: the
//   swap would then compare against the wrong word and the count would fall
//   short of base + harts * iters.
void kernel_cas_counter(kernel_arg_t* __UNIFORM__ arg) {
  auto counter = (uint32_t*)arg->shared_addr;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    uint32_t expected = __atomic_load_n(counter, __ATOMIC_RELAXED);
    while (!__atomic_compare_exchange_n(counter, &expected, expected + 1, 0,
                                        __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
    }
  }
}

// 18) Compare-and-swap increment on local memory.
//   Test 17 on a group's local-memory word, so the LMEM banks resolve the swap.
//   Each group's leader publishes how far its counter moved past the base, and
//   the published counts sum to harts * iters whatever the launch shape.
void kernel_cas_counter_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto counter  = (uint32_t*)__local_mem();
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  if (threadIdx.x == 0) {
    *counter = CAS_COUNTER_BASE;
  }
  __syncthreads();

  for (uint32_t i = 0; i < arg->iters; ++i) {
    uint32_t expected = __atomic_load_n(counter, __ATOMIC_RELAXED);
    while (!__atomic_compare_exchange_n(counter, &expected, expected + 1, 0,
                                        __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    per_hart[hart_id()] = *counter - CAS_COUNTER_BASE;
  }
}
#endif

#if AMO64_TESTS
// 64-bit atomic primitives. A 64-bit hart has the .D forms natively; a 32-bit
// hart names even-aligned register pairs ("R"), which the assembler only spells
// for amocas.d, so the read-modify-write forms are emitted as raw encodings
// (funct7 = funct5 << 2).
#if VX_CFG_XLEN == 32
#define AMO64_RMW(name, funct7)                                          \
  static inline uint64_t name(volatile uint64_t* p, uint64_t v) {        \
    uint64_t old;                                                        \
    asm volatile (".insn r 0x2f, 3, " #funct7 ", %0, %1, %2"             \
                  : "=R"(old) : "r"(p), "R"(v) : "memory");              \
    return old;                                                          \
  }
AMO64_RMW(amo64_add,  0x00)
AMO64_RMW(amo64_swap, 0x04)
AMO64_RMW(amo64_xor,  0x10)
AMO64_RMW(amo64_or,   0x20)
AMO64_RMW(amo64_and,  0x30)
AMO64_RMW(amo64_min,  0x40)
AMO64_RMW(amo64_max,  0x50)
AMO64_RMW(amo64_minu, 0x60)
AMO64_RMW(amo64_maxu, 0x70)
#if VX_CFG_EXT_ZACAS_ENABLED
// Returns the old value; the swap landed when it equals `expected`.
static inline uint64_t amo64_cas(volatile uint64_t* p, uint64_t expected, uint64_t v) {
  asm volatile ("amocas.d %0, %2, (%1)"
                : "+R"(expected) : "r"(p), "R"(v) : "memory");
  return expected;
}
#endif
#else
#define AMO64_RMW(name, insn)                                            \
  static inline uint64_t name(volatile uint64_t* p, uint64_t v) {        \
    uint64_t old;                                                        \
    asm volatile (insn " %0, %2, (%1)"                                   \
                  : "=r"(old) : "r"(p), "r"(v) : "memory");              \
    return old;                                                          \
  }
AMO64_RMW(amo64_add,  "amoadd.d")
AMO64_RMW(amo64_swap, "amoswap.d")
AMO64_RMW(amo64_xor,  "amoxor.d")
AMO64_RMW(amo64_or,   "amoor.d")
AMO64_RMW(amo64_and,  "amoand.d")
AMO64_RMW(amo64_min,  "amomin.d")
AMO64_RMW(amo64_max,  "amomax.d")
AMO64_RMW(amo64_minu, "amominu.d")
AMO64_RMW(amo64_maxu, "amomaxu.d")
#if VX_CFG_EXT_ZACAS_ENABLED
static inline uint64_t amo64_cas(volatile uint64_t* p, uint64_t expected, uint64_t v) {
  asm volatile ("amocas.d %0, %2, (%1)"
                : "+r"(expected) : "r"(p), "r"(v) : "memory");
  return expected;
}
#endif
#endif

// A value an add counter can hold after fewer than `total` steps from
// AMO64_ADD_BASE (see common.h).
static inline bool amo64_add_valid(uint64_t v, uint32_t total) {
  uint32_t k  = (uint32_t)v - (uint32_t)AMO64_ADD_BASE;
  uint32_t hi = k + ((k >= (uint32_t)(0u - (uint32_t)AMO64_ADD_BASE)) ? 1u : 0u);
  return (k < total) && ((uint32_t)(v >> 32) == hi);
}

static inline bool amo64_minmax_valid(uint64_t v, uint64_t init, uint32_t n) {
  uint32_t k = (uint32_t)v - 1;
  return (v == init) || ((k < n) && (amo64_minmax_value(k) == v));
}

static inline bool amo64_swap_valid(uint64_t v, uint32_t n) {
  uint32_t k = (uint32_t)(v >> 32) - 1;
  return (v == AMO64_SWAP_INIT) || ((k < n) && (amo64_swap_value(k) == v));
}

// Arithmetic atomics over five words: add, signed max/min, unsigned max/min.
// Every hart hammers every word; each returned old value must be one the word
// can hold, and the count of those that are not is the hart's error count.
static uint32_t amo64_arith(volatile uint64_t* w, uint32_t hid, uint32_t n,
                            uint32_t add_total, uint32_t iters) {
  const uint64_t v = amo64_minmax_value(hid);
  uint32_t bad = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    bad += !amo64_add_valid(amo64_add(&w[0], AMO64_ADD_STEP), add_total);
    bad += !amo64_minmax_valid(amo64_max(&w[1], v),  (uint64_t)INT64_MIN, n);
    bad += !amo64_minmax_valid(amo64_min(&w[2], v),  (uint64_t)INT64_MAX, n);
    bad += !amo64_minmax_valid(amo64_maxu(&w[3], v), 0, n);
    bad += !amo64_minmax_valid(amo64_minu(&w[4], v), UINT64_MAX, n);
  }
  return bad;
}

static void amo64_arith_init(volatile uint64_t* w) {
  w[0] = AMO64_ADD_BASE;
  w[1] = (uint64_t)INT64_MIN;
  w[2] = (uint64_t)INT64_MAX;
  w[3] = 0;
  w[4] = UINT64_MAX;
}

// Bitwise atomics and exchange over four words: or, and, xor, swap.
static uint32_t amo64_bits(volatile uint64_t* w, uint32_t hid, uint32_t n, uint32_t iters) {
  const uint64_t b = amo64_bit_value(hid);
  const uint64_t s = amo64_swap_value(hid);
  uint32_t bad = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    amo64_or(&w[0], b);
    amo64_and(&w[1], ~b);
    amo64_xor(&w[2], b);
    bad += !amo64_swap_valid(amo64_swap(&w[3], s), n);
  }
  // an odd number of xors per hart, so the result is the xor of every operand
  if ((iters & 1) == 0) {
    amo64_xor(&w[2], b);
  }
  return bad;
}

static void amo64_bits_init(volatile uint64_t* w) {
  w[0] = 0;
  w[1] = UINT64_MAX;
  w[2] = AMO64_XOR_INIT;
  w[3] = AMO64_SWAP_INIT;
}

// 64-bit arithmetic atomics on global memory, every hart on the same words.
void kernel_amo64_arith(kernel_arg_t* __UNIFORM__ arg) {
  auto w        = (volatile uint64_t*)arg->shared_addr;
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  per_hart[hid] = amo64_arith(w, hid, arg->num_harts, arg->num_harts * arg->iters, arg->iters);
}

// The same on local memory. The host cannot read it, so each group's leader
// publishes the group's final words, tagged with the group size, into its own
// line of the shared region.
void kernel_amo64_arith_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto w        = (volatile uint64_t*)__local_mem();
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  auto out      = (volatile uint64_t*)arg->shared_addr;
  const uint32_t hid = hart_id();
  if (threadIdx.x == 0) {
    amo64_arith_init(w);
  }
  __syncthreads();
  per_hart[hid] = amo64_arith(w, hid, arg->num_harts, blockDim.x * arg->iters, arg->iters);
  __syncthreads();
  if (threadIdx.x == 0) {
    for (uint32_t i = 0; i < 5; ++i) {
      out[hid * 8 + i] = w[i];
    }
    out[hid * 8 + 7] = AMO64_LMEM_MARK | blockDim.x;
  }
}

void kernel_amo64_bits(kernel_arg_t* __UNIFORM__ arg) {
  auto w        = (volatile uint64_t*)arg->shared_addr;
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  per_hart[hid] = amo64_bits(w, hid, arg->num_harts, arg->iters);
}

void kernel_amo64_bits_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto w        = (volatile uint64_t*)__local_mem();
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  auto out      = (volatile uint64_t*)arg->shared_addr;
  const uint32_t hid = hart_id();
  if (threadIdx.x == 0) {
    amo64_bits_init(w);
  }
  __syncthreads();
  per_hart[hid] = amo64_bits(w, hid, arg->num_harts, arg->iters);
  __syncthreads();
  if (threadIdx.x == 0) {
    for (uint32_t i = 0; i < 4; ++i) {
      out[hid * 8 + i] = w[i];
    }
    out[hid * 8 + 7] = AMO64_LMEM_MARK | blockDim.x;
  }
}

#if VX_CFG_EXT_ZACAS_ENABLED
// Compare-and-swap increment of a 64-bit counter whose low word wraps. The
// initial read is two loads on a 32-bit hart and may be torn; the swap then
// fails, returns the word it saw, and the retry carries on from there.
static void amo64_cas_count(volatile uint64_t* counter, uint32_t iters) {
  for (uint32_t i = 0; i < iters; ++i) {
    uint64_t expected = *counter;
    for (;;) {
      uint64_t seen = amo64_cas(counter, expected, expected + 1);
      if (seen == expected) {
        break;
      }
      expected = seen;
    }
  }
}

void kernel_amo64_cas_counter(kernel_arg_t* __UNIFORM__ arg) {
  amo64_cas_count((volatile uint64_t*)arg->shared_addr, arg->iters);
}

void kernel_amo64_cas_counter_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto counter  = (volatile uint64_t*)__local_mem();
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  if (threadIdx.x == 0) {
    *counter = AMO64_CAS_BASE;
  }
  __syncthreads();
  amo64_cas_count(counter, arg->iters);
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t moved = *counter - AMO64_CAS_BASE;
    // a count that does not fit the low word means the high word went wrong
    per_hart[hart_id()] = (moved >> 32) ? 0xffffffffu : (uint32_t)moved;
  }
}

// Every hart races for each slot of a 64-bit ladder; the per-hart win counts
// check the returned old value, which must equal the comparand exactly once
// per slot.
void kernel_amo64_cas_ladder(kernel_arg_t* __UNIFORM__ arg) {
  auto slots_p  = (volatile uint64_t*)arg->shared_addr;
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  const uint32_t slots = (arg->iters < CAS_LADDER_SLOTS) ? arg->iters : CAS_LADDER_SLOTS;
  uint32_t wins = 0;
  for (uint32_t i = 0; i < slots; ++i) {
    if (amo64_cas(&slots_p[i], 0, amo64_claim_value(hid)) == 0) {
      ++wins;
    }
  }
  per_hart[hid] = wins;
}
#endif
#endif

// --- 64-bit atomic ordering hazards (register pairs on a 32-bit hart) -------
#if AMO64_TESTS && (VX_CFG_XLEN == 32)
// B1 (write-after-write on rd+1). A 64-bit atomic writes rd and rd+1; the LSU
// reserves rd+1 at issue (the AMO_HI uop) and retires it from the atomic's old
// high word. If the burst does not also read rd+1 before issuing, an older,
// still-pending write of rd+1 -- a slow load that missed in the cache -- lands
// after the reservation and either clobbers the high word or re-reserves rd+1
// forever (a hang). Here a9 (== the amoadd.d's rd+1, a8/a9 the rd pair) is
// written by a slow load one instruction before the atomic, so the atomic's
// old high word and a9 collide. The harness verifies both the final counter
// and that every returned old value was decodable.
static uint32_t amo64_war(volatile uint64_t* w, volatile uint32_t* slow,
                          uint32_t total, uint32_t iters) {
  uint32_t bad = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    register uint64_t acc asm("a6"); // rd pair a6/a7
    register uintptr_t wp asm("a4") = (uintptr_t)w;
    register uint64_t step asm("a2") = AMO64_ADD_STEP; // operand pair a2/a3
    register uintptr_t sp asm("a5") = (uintptr_t)slow;
    // a7 (== the amoadd.d's rd+1) is written by a cache-missing load one
    // instruction before the atomic. The burst reads rd+1 as its comparand
    // stage, which orders the atomic behind that load; without it the load's
    // late landing clobbers acc's high word (or re-reserves a7 and hangs).
    asm volatile ("lw a7, 0(a5)\n"
                  ".insn r 0x2f, 3, 0x00, a6, a4, a2"
                  : "=r"(acc)
                  : "r"(wp), "r"(step), "r"(sp)
                  : "a7", "memory");
    bad += !amo64_add_valid(acc, total);
  }
  return bad;
}

// B1 on global memory, every hart on the same counter. The per-hart slow word
// lives in the hart's own 64B line of the shared region (offset 16), clear of
// the counter in hart 0's line.
void kernel_amo64_war(kernel_arg_t* __UNIFORM__ arg) {
  auto w        = (volatile uint64_t*)arg->shared_addr;
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  auto slow     = (volatile uint32_t*)arg->shared_addr;
  const uint32_t hid = hart_id();
  per_hart[hid] = amo64_war(w, &slow[hid * 16 + 4], arg->num_harts * arg->iters, arg->iters);
}

// B1 with the atomic on local memory: its response returns long before the
// older global load that missed, which is the window the burst must close.
void kernel_amo64_war_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto w        = (volatile uint64_t*)__local_mem();
  auto slow     = (volatile uint32_t*)arg->per_hart_addr;
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  auto out      = (volatile uint64_t*)arg->shared_addr;
  const uint32_t hid = hart_id();
  if (threadIdx.x == 0) {
    w[0] = AMO64_ADD_BASE;
  }
  __syncthreads();
  // the slow word is the hart's own result slot, zero until it is written below
  uint32_t bad = amo64_war(w, &slow[hid], blockDim.x * arg->iters, arg->iters);
  per_hart[hid] = bad;
  __syncthreads();
  if (threadIdx.x == 0) {
    out[hid * 8]     = w[0];
    out[hid * 8 + 7] = AMO64_LMEM_MARK | blockDim.x;
  }
}

// B2 (a thread's own store must precede its later atomic to the same word).
// Each thread stores 0 to its private 64-bit word, then does a 64-bit atomic
// add of 1 to it. Both must travel the thread's own lane so the atomic cannot
// overtake the store; the result must be 1. Threads t>=2 are the ones a
// lane-0/1 reroute would reorder, and bank contention (every thread on its own
// word in the same local-memory banks) opens the window.
void kernel_amo64_store_amo_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto base     = (volatile uint64_t*)__local_mem();
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  volatile uint64_t* w = &base[threadIdx.x];
  uint32_t bad = 0;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    *w = 0;
    uint64_t old = amo64_add(w, 1);
    bad += (old != 0) + ((*w) != 1);
  }
  per_hart[hid] = bad;
}

// B2 on global memory. A warp's threads take words one bank stride apart, so
// the warp's store request queues on one dcache bank lane by lane while the
// thread's later atomic would have a free port if it left the thread's lane.
#define AMO64_B2_STRIDE (VX_CFG_DCACHE_NUM_BANKS * VX_CFG_L1_LINE_SIZE / 8)
void kernel_amo64_store_amo(kernel_arg_t* __UNIFORM__ arg) {
  auto base     = (volatile uint64_t*)arg->shared_addr;
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  // a multiple of the bank stride that leaves room for every warp's word
  const uint32_t warps = arg->num_harts / VX_CFG_NUM_THREADS;
  uint32_t stride = AMO64_B2_STRIDE;
  while (stride < warps) {
    stride += AMO64_B2_STRIDE;
  }
  volatile uint64_t* w = &base[(hid % VX_CFG_NUM_THREADS) * stride
                               + hid / VX_CFG_NUM_THREADS];
  uint32_t bad = 0;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    *w = 0;
    uint64_t old = amo64_add(w, 1);
    bad += (old != 0) + ((*w) != 1);
  }
  per_hart[hid] = bad;
}
#endif

#if VX_CFG_XLEN == 32
// Encodings a 32-bit hart must refuse with an illegal-instruction trap: LR.D,
// which has no register-pair form, and a .D atomic naming an odd register (with
// AMO64) or any .D atomic (without it). The handler records mcause in t1 and
// resumes after the instruction; the word the instructions name must not change.
extern "C" void amo64_illegal_handler();
asm (".align 4\n"
     "amo64_illegal_handler:\n"
     "  csrr t0, mepc\n"
     "  addi t0, t0, 4\n"
     "  csrw mepc, t0\n"
     "  csrr t1, mcause\n"
     "  mret\n");

void kernel_amo64_illegal(kernel_arg_t* __UNIFORM__ arg) {
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  uintptr_t saved_mtvec;
  asm volatile ("csrr %0, mtvec" : "=r"(saved_mtvec));
  asm volatile ("csrw mtvec, %0" :: "r"(&amo64_illegal_handler));
  register uintptr_t addr asm("a4") = (uintptr_t)arg->shared_addr;
  // LR.D has no register-pair form.
  register uint32_t cause_lrd asm("t1") = 0;
  asm volatile (".insn r 0x2f, 3, 0x08, a2, a4, x0" // lr.d
                : "+r"(cause_lrd) : "r"(addr) : "t0", "a2", "a3", "memory");
  uint32_t c_lrd = cause_lrd;
#if VX_CFG_EXT_AMO64_ENABLED
  // With AMO64 a .D atomic naming an odd register in either pair is illegal.
  register uint32_t cause_ord asm("t1") = 0;
  asm volatile (".insn r 0x2f, 3, 0x00, a3, a4, a6" // amoadd.d, odd rd
                : "+r"(cause_ord) : "r"(addr) : "t0", "a3", "memory");
  register uint32_t cause_ors asm("t1") = 0;
  asm volatile (".insn r 0x2f, 3, 0x00, a2, a4, a5" // amoadd.d, odd rs2
                : "+r"(cause_ors) : "r"(addr) : "t0", "a2", "a3", "memory");
  uint32_t c2 = cause_ord, c3 = cause_ors;
#else
  // Without AMO64 every .D atomic is illegal.
  register uint32_t cause_addd asm("t1") = 0;
  asm volatile (".insn r 0x2f, 3, 0x00, a2, a4, a6" // amoadd.d
                : "+r"(cause_addd) : "r"(addr) : "t0", "a2", "a3", "memory");
  uint32_t c2 = cause_addd, c3 = cause_addd;
#endif
  asm volatile ("csrw mtvec, %0" :: "r"(saved_mtvec));
  per_hart[hart_id()] = c_lrd | (c2 << 8) | (c3 << 16);
}
#endif

static const PFN_Kernel sc_tests[] = {
  kernel_amoadd,             // 0
  kernel_amoor,              // 1
  kernel_amoand,             // 2
  kernel_amoxor,             // 3
  kernel_amomax,             // 4
  kernel_amominu,            // 5
  kernel_amoswap,            // 6
  kernel_lrsc_counter,       // 7
  kernel_amoadd_aqrl,        // 8
  kernel_lrsc_counter_aqrl,  // 9
  kernel_atomic_reduction,   // 10  CUDA-style reduction
  kernel_atomic_critical,    // 11  CUDA-style critical section
  kernel_self_consistency,   // 12  issuer self-consistency (load->AMO->load)
  kernel_amoadd_lmem,        // 13  AMOADD on local memory (LMEM banks)
  kernel_lrsc_lmem,          // 14  LR/SC on local memory (LMEM banks)
#if VX_CFG_EXT_ZACAS_ENABLED
  kernel_cas_ladder,         // 15  native compare-and-swap (Zacas amocas)
  kernel_cas_lmem,           // 16  compare-and-swap on local memory (LMEM banks)
  kernel_cas_counter,        // 17  compare-and-swap increment, nonzero comparands
  kernel_cas_counter_lmem,   // 18  the same on local memory (LMEM banks)
#endif
#if AMO64_TESTS
  kernel_amo64_arith,        // 64-bit add/min/max (global)
  kernel_amo64_arith_lmem,   // the same on local memory
  kernel_amo64_bits,         // 64-bit or/and/xor/swap (global)
  kernel_amo64_bits_lmem,    // the same on local memory
#if VX_CFG_EXT_ZACAS_ENABLED
  kernel_amo64_cas_counter,      // 64-bit compare-and-swap counter (global)
  kernel_amo64_cas_counter_lmem, // the same on local memory
  kernel_amo64_cas_ladder,       // 64-bit compare-and-swap ladder
#endif
#if VX_CFG_XLEN == 32
  kernel_amo64_war,              // write-after-write on rd+1 (global)
  kernel_amo64_war_lmem,         // the same on local memory
  kernel_amo64_store_amo,        // a thread's store orders before its atomic
  kernel_amo64_store_amo_lmem,   // the same on local memory
#endif
#endif
#if VX_CFG_XLEN == 32
  kernel_amo64_illegal,      // refused .D encodings trap
#endif
};

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  sc_tests[arg->testid](arg);
}
