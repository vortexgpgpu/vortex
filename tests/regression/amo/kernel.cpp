#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// All harts contend on a single shared word. Each test exercises a
// different RVA primitive; the host validates the closed-form expected
// final value against the post-run shared word, plus per-hart "observed
// old value" / "iterations completed" counters where applicable.

typedef void (*PFN_Kernel)(kernel_arg_t* __UNIFORM__ arg);

// Globally-unique hart id matching the simulator's make_hart_id():
// (cid * NUM_WARPS + wid) * NUM_THREADS + tid.
static inline uint32_t hart_id() {
  return (vx_core_id() * vx_num_warps() + vx_warp_id()) * vx_num_threads()
       + vx_thread_id();
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

// 10) Atomic reduction (textbook CUDA atomicAdd reduction).
//   Pattern (CUDA Programming Guide, "Reduction" example):
//     __global__ void reduce(int* in, int* out) {
//       int i = blockIdx.x*blockDim.x + threadIdx.x;
//       atomicAdd(out, in[i]);
//     }
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

// 11) Atomic critical section (textbook CUDA atomicCAS spinlock).
//   Pattern (CUDA Best Practices, classic atomicCAS lock):
//     while (atomicCAS(&lock, 0, 1) != 0) ;  // acquire
//     // ... non-atomic critical section ...
//     atomicExch(&lock, 0);                  // release
//   The RV equivalent uses amoswap.w to atomically swap 1 in and read
//   back the prior value; success when prior was 0.
//
//   IMPORTANT — SIMD-lockstep gate: this pattern deadlocks within a
//   warp on architectures with strict warp-lockstep execution (Vortex,
//   pre-Volta CUDA): one thread acquires the lock and diverges into
//   the critical section, but the IPDOM stack stalls it until the
//   other threads in the warp leave the else-branch spin loop — which
//   they can't, because that requires the holder to release. The
//   well-known workaround (also documented in CUDA Best Practices) is
//   to serialize at warp granularity: only one thread per warp
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
};

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  sc_tests[arg->testid](arg);
}
