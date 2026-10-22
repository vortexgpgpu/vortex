#pragma once

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vortex.h>
#include <VX_types.h>
#include "common.h"

void cleanup();

#define RT_CHECK(_expr)                                       \
   do {                                                       \
     int _ret = _expr;                                        \
     if (0 == _ret) break;                                    \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
     cleanup();                                               \
     exit(-1);                                                \
   } while (false)

class ITestCase;

class TestSuite {
public:
  TestSuite(vx_device_h device);
  ~TestSuite();

  ITestCase* get_test(int testid) const { return _tests.at(testid); }
  void add_test(ITestCase* t) { _tests.push_back(t); }
  size_t size() const { return _tests.size(); }
  vx_device_h device() const { return device_; }

private:
  std::vector<ITestCase*> _tests;
  vx_device_h device_;
};

class ITestCase {
public:
  ITestCase(TestSuite* suite, const char* name) : suite_(suite), name_(name) {}
  virtual ~ITestCase() {}

  TestSuite* suite() const { return suite_; }
  const char* name() const { return name_; }

  // Initialize the shared word and per-hart buffer to test-specific
  // initial values. `n` is num_harts (size of per_hart buffer in u32s).
  virtual void setup(uint32_t n, uint32_t* shared, uint32_t* per_hart) = 0;

  // Validate the post-run shared word and per-hart buffer.
  // Returns 0 on pass, error count on fail.
  virtual int verify(uint32_t n, uint32_t iters,
                     const uint32_t* shared, const uint32_t* per_hart) = 0;

protected:
  TestSuite* suite_;
  const char* const name_;
};

// 0) AMOADD hammer: each hart does +1 iter times. Final = n * iters.
class Test_AMOADD : public ITestCase {
public:
  Test_AMOADD(TestSuite* s) : ITestCase(s, "amoadd") {}
  void setup(uint32_t, uint32_t* shared, uint32_t* /*ph*/) override {
    *shared = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* /*ph*/) override {
    uint32_t expected = n * iters;
    if (*shared != expected) {
      std::cout << "  shared=" << *shared << " expected=" << expected << std::endl;
      return 1;
    }
    return 0;
  }
};

// 1) AMOOR hammer: each hart sets bit (hart_id % 32) idempotently.
//    Final = OR of all such bits = 0xFFFFFFFF for n>=32 (all bits 0..31 hit),
//    else (1 << n) - 1.
class Test_AMOOR : public ITestCase {
public:
  Test_AMOOR(TestSuite* s) : ITestCase(s, "amoor") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override { *shared = 0; }
  int verify(uint32_t n, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t*) override {
    uint32_t expected = 0;
    for (uint32_t h = 0; h < n; ++h) expected |= 1u << (h & 31);
    if (*shared != expected) {
      std::cout << "  shared=0x" << std::hex << *shared
                << " expected=0x" << expected << std::dec << std::endl;
      return 1;
    }
    return 0;
  }
};

// 2) AMOAND hammer: start all-ones, each hart clears bit (hart_id % 32).
class Test_AMOAND : public ITestCase {
public:
  Test_AMOAND(TestSuite* s) : ITestCase(s, "amoand") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    *shared = 0xffffffff;
  }
  int verify(uint32_t n, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t*) override {
    uint32_t cleared = 0;
    for (uint32_t h = 0; h < n; ++h) cleared |= 1u << (h & 31);
    uint32_t expected = ~cleared;
    if (*shared != expected) {
      std::cout << "  shared=0x" << std::hex << *shared
                << " expected=0x" << expected << std::dec << std::endl;
      return 1;
    }
    return 0;
  }
};

// 3) AMOXOR hammer: each hart toggles its bit twice per iter. Final = initial.
class Test_AMOXOR : public ITestCase {
public:
  Test_AMOXOR(TestSuite* s) : ITestCase(s, "amoxor") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    *shared = 0xa5a5a5a5;
  }
  int verify(uint32_t /*n*/, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t*) override {
    if (*shared != 0xa5a5a5a5) {
      std::cout << "  shared=0x" << std::hex << *shared
                << " expected=0xa5a5a5a5" << std::dec << std::endl;
      return 1;
    }
    return 0;
  }
};

// 4) AMOMAX hammer (signed): each hart writes its hart_id. Final = n - 1.
class Test_AMOMAX : public ITestCase {
public:
  Test_AMOMAX(TestSuite* s) : ITestCase(s, "amomax") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    // INT32_MIN as initial so any unsigned hart_id wins.
    *shared = 0x80000000u;
  }
  int verify(uint32_t n, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t*) override {
    int32_t got = (int32_t)*shared;
    int32_t expected = (int32_t)(n - 1);
    if (got != expected) {
      std::cout << "  shared=" << got << " expected=" << expected << std::endl;
      return 1;
    }
    return 0;
  }
};

// 5) AMOMINU hammer (unsigned): each hart writes its hart_id (≥ 0).
//    Final = 0 (the lowest hart_id always wins).
class Test_AMOMINU : public ITestCase {
public:
  Test_AMOMINU(TestSuite* s) : ITestCase(s, "amominu") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    *shared = 0xffffffffu;
  }
  int verify(uint32_t /*n*/, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t*) override {
    if (*shared != 0) {
      std::cout << "  shared=0x" << std::hex << *shared
                << " expected=0" << std::dec << std::endl;
      return 1;
    }
    return 0;
  }
};

// 6) AMOSWAP exchange: each hart swaps in its hart_id; per-hart obs[]
//    captures last observed-old value. Validate:
//    - final shared is some hart_id (0..n-1) — some hart wrote it last.
//    - every observed value is either initial sentinel or a valid hart_id.
class Test_AMOSWAP : public ITestCase {
public:
  Test_AMOSWAP(TestSuite* s) : ITestCase(s, "amoswap") {}
  void setup(uint32_t n, uint32_t* shared, uint32_t* per_hart) override {
    *shared = 0xdeadbeefu; // sentinel (not a hart_id)
    for (uint32_t i = 0; i < n; ++i) per_hart[i] = 0xdeadbeefu;
  }
  int verify(uint32_t n, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t* per_hart) override {
    int errors = 0;
    if (*shared >= n) {
      std::cout << "  final shared=" << *shared
                << " not a valid hart_id (n=" << n << ")" << std::endl;
      ++errors;
    }
    for (uint32_t h = 0; h < n; ++h) {
      uint32_t obs = per_hart[h];
      bool ok = (obs == 0xdeadbeefu) || (obs < n);
      if (!ok) {
        std::cout << "  hart " << h << " observed garbage=0x" << std::hex
                  << obs << std::dec << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

// 7) LR/SC counter (lock-free): final = n * iters.
class Test_LRSC_COUNTER : public ITestCase {
public:
  Test_LRSC_COUNTER(TestSuite* s) : ITestCase(s, "lrsc_counter") {}
  void setup(uint32_t n, uint32_t* shared, uint32_t* per_hart) override {
    *shared = 0;
    for (uint32_t i = 0; i < n; ++i) per_hart[i] = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* /*ph*/) override {
    uint32_t expected = n * iters;
    if (*shared != expected) {
      std::cout << "  shared=" << *shared << " expected=" << expected << std::endl;
      return 1;
    }
    return 0;
  }
};

// 8) AMOADD.W.AQRL hammer: same expected as Test_AMOADD. Validates
//    the .aqrl encoding path; under SimX's SC ordering the observable
//    result is identical to the plain variant.
class Test_AMOADD_AQRL : public ITestCase {
public:
  Test_AMOADD_AQRL(TestSuite* s) : ITestCase(s, "amoadd_aqrl") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override { *shared = 0; }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t*) override {
    uint32_t expected = n * iters;
    if (*shared != expected) {
      std::cout << "  shared=" << *shared << " expected=" << expected << std::endl;
      return 1;
    }
    return 0;
  }
};

// 9) LR.W.AQ / SC.W.RL counter: same expected as Test_LRSC_COUNTER.
//    Validates the .aq / .rl encoding paths.
class Test_LRSC_COUNTER_AQRL : public ITestCase {
public:
  Test_LRSC_COUNTER_AQRL(TestSuite* s) : ITestCase(s, "lrsc_counter_aqrl") {}
  void setup(uint32_t /*n*/, uint32_t* shared, uint32_t*) override {
    *shared = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t*) override {
    uint32_t expected = n * iters;
    if (*shared != expected) {
      std::cout << "  shared=" << *shared << " expected=" << expected << std::endl;
      return 1;
    }
    return 0;
  }
};

// 10) Atomic reduction (CUDA atomicAdd reduction). per_hart[h] = h+1
//     supplied by host; kernel atomicAdds each into shared. Expected
//     final = sum_{h=0..n-1} (h+1) = n*(n+1)/2.
class Test_ATOMIC_REDUCTION : public ITestCase {
public:
  Test_ATOMIC_REDUCTION(TestSuite* s) : ITestCase(s, "atomic_reduction") {}
  void setup(uint32_t n, uint32_t* shared, uint32_t* per_hart) override {
    *shared = 0;
    for (uint32_t i = 0; i < n; ++i) per_hart[i] = i + 1;
  }
  int verify(uint32_t n, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t*) override {
    uint32_t expected = n * (n + 1) / 2;
    if (*shared != expected) {
      std::cout << "  shared=" << *shared << " expected=" << expected << std::endl;
      return 1;
    }
    return 0;
  }
};

// 11) Atomic critical section (CUDA atomicCAS spinlock). Only thread 0
//     of each warp participates. Active harts = num_cores * num_warps
//     = num_harts / num_threads. Final counter = active * iters; lock must end at 0.
class Test_ATOMIC_CRITICAL : public ITestCase {
public:
  Test_ATOMIC_CRITICAL(TestSuite* s, uint32_t num_threads_per_warp)
    : ITestCase(s, "atomic_critical")
    , num_threads_(num_threads_per_warp) {}
  void setup(uint32_t /*n*/, uint32_t* shared, uint32_t* per_hart) override {
    *shared = 0;
    per_hart[0] = 0;  // lock starts unheld
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* per_hart) override {
    int errors = 0;
    uint32_t active   = n / num_threads_;          // one thread per warp
    uint32_t expected = active * iters;
    if (*shared != expected) {
      std::cout << "  counter=" << *shared << " expected=" << expected
                << " (active_harts=" << active << ")" << std::endl;
      ++errors;
    }
    if (per_hart[0] != 0) {
      std::cout << "  lock not released, lock=" << per_hart[0] << std::endl;
      ++errors;
    }
    return errors;
  }
private:
  uint32_t num_threads_;
};

// 12) Issuer self-consistency: each hart caches its own private word,
//     atomically increments it, and must observe the increment via a
//     later plain load. Final per_hart[hid] == 1 for every hart. Fails
//     when a non-LLC cache does not invalidate the issuer's stale copy
//     on the AMO (post-AMO plain load returns the pre-AMO value).
class Test_SELF_CONSISTENCY : public ITestCase {
public:
  Test_SELF_CONSISTENCY(TestSuite* s) : ITestCase(s, "self_consistency") {}
  void setup(uint32_t n, uint32_t* /*shared*/, uint32_t* per_hart) override {
    for (uint32_t h = 0; h < n; ++h) per_hart[h] = 0;
  }
  int verify(uint32_t n, uint32_t /*iters*/,
             const uint32_t* /*shared*/, const uint32_t* per_hart) override {
    int errors = 0;
    for (uint32_t h = 0; h < n; ++h) {
      if (per_hart[h] != 1) {
        if (errors < 4)
          std::cout << "  hart " << h << " observed " << per_hart[h]
                    << " expected 1 (stale post-AMO load)" << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

// 13) LMEM AMOADD hammer: the same contention as amoadd, on local memory, so
//     the LMEM banks resolve the read-modify-write rather than the dcache.
//     Local memory is per-workgroup, so each group's leader publishes its
//     group's final count and the published values sum to n * iters -- a closed
//     form independent of how the launch was shaped into groups.
class Test_AMOADD_LMEM : public ITestCase {
public:
  Test_AMOADD_LMEM(TestSuite* s) : ITestCase(s, "amoadd_lmem") {}
  void setup(uint32_t n, uint32_t* /*shared*/, uint32_t* per_hart) override {
    for (uint32_t h = 0; h < n; ++h) per_hart[h] = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* /*shared*/, const uint32_t* per_hart) override {
    uint64_t total = 0;
    uint32_t groups = 0;
    for (uint32_t h = 0; h < n; ++h) {
      if (per_hart[h] != 0) { total += per_hart[h]; ++groups; }
    }
    uint64_t expected = (uint64_t)n * iters;
    if (groups == 0) {
      std::cout << "  no group published a count -- the kernel did not run"
                << std::endl;
      return 1;
    }
    if (total != expected) {
      std::cout << "  published total=" << total << " expected=" << expected
                << " across " << groups << " group(s)" << std::endl;
      return 1;
    }
    return 0;
  }
};

// 14) LMEM LR/SC counter: the lock-free retry of test 7 on local memory, so
//     the LMEM banks resolve the reservation. Published per-group counts sum
//     to n * iters, independent of how the launch was shaped into groups.
class Test_LRSC_LMEM : public ITestCase {
public:
  Test_LRSC_LMEM(TestSuite* s) : ITestCase(s, "lrsc_lmem") {}
  void setup(uint32_t n, uint32_t* /*shared*/, uint32_t* per_hart) override {
    for (uint32_t h = 0; h < n; ++h) per_hart[h] = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* /*shared*/, const uint32_t* per_hart) override {
    uint64_t total = 0;
    uint32_t groups = 0;
    for (uint32_t h = 0; h < n; ++h) {
      if (per_hart[h] != 0) { total += per_hart[h]; ++groups; }
    }
    uint64_t expected = (uint64_t)n * iters;
    if (groups == 0) {
      std::cout << "  no group published a count -- no store-conditional succeeded"
                << std::endl;
      return 1;
    }
    if (total != expected) {
      std::cout << "  published total=" << total << " expected=" << expected
                << " across " << groups << " group(s)" << std::endl;
      return 1;
    }
    return 0;
  }
};

// 15) Native compare-and-swap ladder: every hart races for each slot, so each
//     slot has exactly one winner. Summing the per-hart win counts checks the
//     per-lane return value, which a compare-and-swap that returned the wrong
//     old word would get wrong even while leaving the ladder itself plausible.
class Test_CAS_LADDER : public ITestCase {
public:
  Test_CAS_LADDER(TestSuite* s) : ITestCase(s, "cas_ladder") {}
  void setup(uint32_t n, uint32_t* shared, uint32_t* per_hart) override {
    for (uint32_t i = 0; i < n; ++i) per_hart[i] = 0;
    // The ladder lives in the shared region; iters slots, all unclaimed.
    for (uint32_t i = 0; i < CAS_LADDER_SLOTS; ++i) shared[i] = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* per_hart) override {
    const uint32_t slots = (iters < CAS_LADDER_SLOTS) ? iters : CAS_LADDER_SLOTS;
    uint64_t wins = 0;
    for (uint32_t h = 0; h < n; ++h) wins += per_hart[h];
    if (wins != slots) {
      std::cout << "  wins=" << wins << " expected=" << slots
                << " (each slot must have exactly one winner)" << std::endl;
      return 1;
    }
    // Every slot must hold some hart's claim, and no slot may still be empty.
    for (uint32_t i = 0; i < slots; ++i) {
      if (shared[i] == 0 || shared[i] > n) {
        std::cout << "  slot " << i << " holds " << shared[i]
                  << ", expected a claim in 1.." << n << std::endl;
        return 1;
      }
    }
    return 0;
  }
};

// 16) LMEM compare-and-swap ladder: the cas_ladder race on local memory, so
//     the LMEM banks resolve the swap and the returned old word. Each group
//     has its own ladder with exactly one winner per slot; the leaders' publish
//     marker carries the group count, so the win counts must sum to
//     groups * slots whatever the launch geometry.
class Test_CAS_LMEM : public ITestCase {
public:
  Test_CAS_LMEM(TestSuite* s) : ITestCase(s, "cas_lmem") {}
  void setup(uint32_t n, uint32_t* /*shared*/, uint32_t* per_hart) override {
    for (uint32_t h = 0; h < n; ++h) per_hart[h] = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* /*shared*/, const uint32_t* per_hart) override {
    const uint32_t slots = (iters < CAS_LMEM_SLOTS) ? iters : CAS_LMEM_SLOTS;
    uint64_t wins = 0;
    uint32_t groups = 0;
    for (uint32_t h = 0; h < n; ++h) {
      uint32_t v = per_hart[h];
      if (v >= CAS_LMEM_GROUP_MARK) { ++groups; v -= CAS_LMEM_GROUP_MARK; }
      wins += v;
    }
    if (groups == 0) {
      std::cout << "  no group published -- the kernel did not run" << std::endl;
      return 1;
    }
    uint64_t expected = (uint64_t)groups * slots;
    if (wins != expected) {
      std::cout << "  wins=" << wins << " expected=" << expected
                << " across " << groups << " group(s)"
                << " (each slot must have exactly one winner)" << std::endl;
      return 1;
    }
    return 0;
  }
};

// 17) Compare-and-swap counter on global memory: each hart adds iters by
//     read/compare-and-swap retries from a nonzero base, so the final word is
//     base + n * iters only if every swap compared against the word it read.
class Test_CAS_COUNTER : public ITestCase {
public:
  Test_CAS_COUNTER(TestSuite* s) : ITestCase(s, "cas_counter") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    *shared = CAS_COUNTER_BASE;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t*) override {
    const uint32_t expected = CAS_COUNTER_BASE + n * iters;
    if (*shared != expected) {
      std::cout << "  shared=0x" << std::hex << *shared << " expected=0x"
                << expected << std::dec << std::endl;
      return 1;
    }
    return 0;
  }
};

// 18) Compare-and-swap counter on local memory: per-group counters from the same
//     nonzero base; the leaders' published increments sum to n * iters.
class Test_CAS_COUNTER_LMEM : public ITestCase {
public:
  Test_CAS_COUNTER_LMEM(TestSuite* s) : ITestCase(s, "cas_counter_lmem") {}
  void setup(uint32_t n, uint32_t* /*shared*/, uint32_t* per_hart) override {
    for (uint32_t h = 0; h < n; ++h) per_hart[h] = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* /*shared*/, const uint32_t* per_hart) override {
    uint64_t total = 0;
    uint32_t groups = 0;
    for (uint32_t h = 0; h < n; ++h) {
      if (per_hart[h] != 0) { total += per_hart[h]; ++groups; }
    }
    const uint64_t expected = (uint64_t)n * iters;
    if (groups == 0) {
      std::cout << "  no group published a count -- the kernel did not run"
                << std::endl;
      return 1;
    }
    if (total != expected) {
      std::cout << "  published total=" << total << " expected=" << expected
                << " across " << groups << " group(s)" << std::endl;
      return 1;
    }
    return 0;
  }
};

#if AMO64_TESTS
static inline uint64_t word64(const uint32_t* p, uint32_t i) {
  return ((uint64_t)p[2 * i + 1] << 32) | p[2 * i];
}

static inline void set_word64(uint32_t* p, uint32_t i, uint64_t v) {
  p[2 * i]     = (uint32_t)v;
  p[2 * i + 1] = (uint32_t)(v >> 32);
}

static int check64(const char* what, uint64_t got, uint64_t expected) {
  if (got == expected) {
    return 0;
  }
  std::cout << "  " << what << "=0x" << std::hex << got << " expected=0x"
            << expected << std::dec << std::endl;
  return 1;
}

static int check_no_bad_olds(uint32_t n, const uint32_t* per_hart) {
  for (uint32_t h = 0; h < n; ++h) {
    if (per_hart[h] != 0) {
      std::cout << "  hart " << h << " saw " << per_hart[h]
                << " old value(s) the word could not hold" << std::endl;
      return 1;
    }
  }
  return 0;
}

struct Amo64Extremes {
  uint64_t max_s, min_s, max_u, min_u, bits_or, bits_xor;
  Amo64Extremes(uint32_t n) {
    int64_t mx = INT64_MIN, mn = INT64_MAX;
    uint64_t mxu = 0, mnu = UINT64_MAX, bo = 0, bx = 0;
    for (uint32_t h = 0; h < n; ++h) {
      uint64_t v = amo64_minmax_value(h);
      mx  = std::max(mx, (int64_t)v);
      mn  = std::min(mn, (int64_t)v);
      mxu = std::max(mxu, v);
      mnu = std::min(mnu, v);
      bo |= amo64_bit_value(h);
      bx ^= amo64_bit_value(h);
    }
    max_s = (uint64_t)mx; min_s = (uint64_t)mn; max_u = mxu; min_u = mnu;
    bits_or = bo; bits_xor = bx;
  }
};

// Leader records published by the local-memory variants: one 64-byte line per
// group, tagged in its last word with the group size.
static std::vector<uint32_t> amo64_lmem_leaders(uint32_t n, const uint32_t* shared) {
  std::vector<uint32_t> leaders;
  for (uint32_t h = 0; h < n; ++h) {
    uint64_t tag = word64(shared, h * 8 + 7);
    if ((tag >> 32) == (AMO64_LMEM_MARK >> 32)) {
      leaders.push_back(h);
    }
  }
  return leaders;
}

static bool amo64_minmax_in_set(uint64_t v, uint32_t n) {
  uint32_t k = (uint32_t)v - 1;
  return (k < n) && (amo64_minmax_value(k) == v);
}

static bool amo64_swap_in_set(uint64_t v, uint32_t n) {
  uint32_t k = (uint32_t)(v >> 32) - 1;
  return (k < n) && (amo64_swap_value(k) == v);
}

// 64-bit add / signed and unsigned min and max, every hart on one set of words.
class Test_AMO64_ARITH : public ITestCase {
public:
  Test_AMO64_ARITH(TestSuite* s) : ITestCase(s, "amo64_arith") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    set_word64(shared, 0, AMO64_ADD_BASE);
    set_word64(shared, 1, (uint64_t)INT64_MIN);
    set_word64(shared, 2, (uint64_t)INT64_MAX);
    set_word64(shared, 3, 0);
    set_word64(shared, 4, UINT64_MAX);
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* per_hart) override {
    Amo64Extremes x(n);
    int err = 0;
    err += check64("add",  word64(shared, 0), AMO64_ADD_BASE + (uint64_t)n * iters * AMO64_ADD_STEP);
    err += check64("max",  word64(shared, 1), x.max_s);
    err += check64("min",  word64(shared, 2), x.min_s);
    err += check64("maxu", word64(shared, 3), x.max_u);
    err += check64("minu", word64(shared, 4), x.min_u);
    err += check_no_bad_olds(n, per_hart);
    return err;
  }
};

class Test_AMO64_ARITH_LMEM : public ITestCase {
public:
  Test_AMO64_ARITH_LMEM(TestSuite* s) : ITestCase(s, "amo64_arith_lmem") {}
  void setup(uint32_t, uint32_t*, uint32_t*) override {}
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* per_hart) override {
    auto leaders = amo64_lmem_leaders(n, shared);
    if (leaders.empty()) {
      std::cout << "  no group published -- the kernel did not run" << std::endl;
      return 1;
    }
    Amo64Extremes x(n);
    int64_t mx = INT64_MIN, mn = INT64_MAX;
    uint64_t mxu = 0, mnu = UINT64_MAX;
    uint32_t harts = 0;
    int err = 0;
    for (auto h : leaders) {
      uint32_t gs = (uint32_t)word64(shared, h * 8 + 7);
      harts += gs;
      err += check64("group add", word64(shared, h * 8 + 0),
                     AMO64_ADD_BASE + (uint64_t)gs * iters * AMO64_ADD_STEP);
      for (uint32_t i = 1; i <= 4; ++i) {
        if (!amo64_minmax_in_set(word64(shared, h * 8 + i), n)) {
          std::cout << "  group of hart " << h << ": word " << i << " = 0x" << std::hex
                    << word64(shared, h * 8 + i) << std::dec << " is no hart's operand" << std::endl;
          ++err;
        }
      }
      mx  = std::max(mx, (int64_t)word64(shared, h * 8 + 1));
      mn  = std::min(mn, (int64_t)word64(shared, h * 8 + 2));
      mxu = std::max(mxu, word64(shared, h * 8 + 3));
      mnu = std::min(mnu, word64(shared, h * 8 + 4));
    }
    if (harts != n) {
      std::cout << "  groups cover " << harts << " harts, expected " << n << std::endl;
      ++err;
    }
    err += check64("max",  (uint64_t)mx, x.max_s);
    err += check64("min",  (uint64_t)mn, x.min_s);
    err += check64("maxu", mxu, x.max_u);
    err += check64("minu", mnu, x.min_u);
    err += check_no_bad_olds(n, per_hart);
    return err;
  }
};

// 64-bit or / and / xor / exchange.
class Test_AMO64_BITS : public ITestCase {
public:
  Test_AMO64_BITS(TestSuite* s) : ITestCase(s, "amo64_bits") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    set_word64(shared, 0, 0);
    set_word64(shared, 1, UINT64_MAX);
    set_word64(shared, 2, AMO64_XOR_INIT);
    set_word64(shared, 3, AMO64_SWAP_INIT);
  }
  int verify(uint32_t n, uint32_t,
             const uint32_t* shared, const uint32_t* per_hart) override {
    Amo64Extremes x(n);
    int err = 0;
    err += check64("or",  word64(shared, 0), x.bits_or);
    err += check64("and", word64(shared, 1), ~x.bits_or);
    err += check64("xor", word64(shared, 2), AMO64_XOR_INIT ^ x.bits_xor);
    if (!amo64_swap_in_set(word64(shared, 3), n)) {
      std::cout << "  swap word = 0x" << std::hex << word64(shared, 3) << std::dec
                << " is no hart's operand" << std::endl;
      ++err;
    }
    err += check_no_bad_olds(n, per_hart);
    return err;
  }
};

class Test_AMO64_BITS_LMEM : public ITestCase {
public:
  Test_AMO64_BITS_LMEM(TestSuite* s) : ITestCase(s, "amo64_bits_lmem") {}
  void setup(uint32_t, uint32_t*, uint32_t*) override {}
  int verify(uint32_t n, uint32_t,
             const uint32_t* shared, const uint32_t* per_hart) override {
    auto leaders = amo64_lmem_leaders(n, shared);
    if (leaders.empty()) {
      std::cout << "  no group published -- the kernel did not run" << std::endl;
      return 1;
    }
    Amo64Extremes x(n);
    uint64_t bo = 0, ba = UINT64_MAX, bx = 0;
    uint32_t harts = 0;
    int err = 0;
    for (auto h : leaders) {
      harts += (uint32_t)word64(shared, h * 8 + 7);
      bo |= word64(shared, h * 8 + 0);
      ba &= word64(shared, h * 8 + 1);
      bx ^= word64(shared, h * 8 + 2) ^ AMO64_XOR_INIT;
      if (!amo64_swap_in_set(word64(shared, h * 8 + 3), n)) {
        std::cout << "  group of hart " << h << ": swap word = 0x" << std::hex
                  << word64(shared, h * 8 + 3) << std::dec << " is no hart's operand" << std::endl;
        ++err;
      }
    }
    if (harts != n) {
      std::cout << "  groups cover " << harts << " harts, expected " << n << std::endl;
      ++err;
    }
    err += check64("or",  bo, x.bits_or);
    err += check64("and", ba, ~x.bits_or);
    err += check64("xor", bx, x.bits_xor);
    err += check_no_bad_olds(n, per_hart);
    return err;
  }
};

#if VX_CFG_EXT_ZACAS_ENABLED
class Test_AMO64_CAS_COUNTER : public ITestCase {
public:
  Test_AMO64_CAS_COUNTER(TestSuite* s) : ITestCase(s, "amo64_cas_counter") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    set_word64(shared, 0, AMO64_CAS_BASE);
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t*) override {
    return check64("counter", word64(shared, 0), AMO64_CAS_BASE + (uint64_t)n * iters);
  }
};

class Test_AMO64_CAS_COUNTER_LMEM : public ITestCase {
public:
  Test_AMO64_CAS_COUNTER_LMEM(TestSuite* s) : ITestCase(s, "amo64_cas_counter_lmem") {}
  void setup(uint32_t, uint32_t*, uint32_t*) override {}
  int verify(uint32_t n, uint32_t iters,
             const uint32_t*, const uint32_t* per_hart) override {
    uint64_t total = 0;
    uint32_t groups = 0;
    for (uint32_t h = 0; h < n; ++h) {
      if (per_hart[h] == 0xffffffffu) {
        std::cout << "  group of hart " << h << " ended with a corrupt high word" << std::endl;
        return 1;
      }
      if (per_hart[h] != 0) { total += per_hart[h]; ++groups; }
    }
    if (groups == 0) {
      std::cout << "  no group published a count -- the kernel did not run" << std::endl;
      return 1;
    }
    if (total != (uint64_t)n * iters) {
      std::cout << "  published total=" << total << " expected=" << (uint64_t)n * iters << std::endl;
      return 1;
    }
    return 0;
  }
};

class Test_AMO64_CAS_LADDER : public ITestCase {
public:
  Test_AMO64_CAS_LADDER(TestSuite* s) : ITestCase(s, "amo64_cas_ladder") {}
  void setup(uint32_t, uint32_t*, uint32_t*) override {}
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* per_hart) override {
    const uint32_t slots = (iters < CAS_LADDER_SLOTS) ? iters : CAS_LADDER_SLOTS;
    uint64_t wins = 0;
    for (uint32_t h = 0; h < n; ++h) {
      wins += per_hart[h];
    }
    if (wins != slots) {
      std::cout << "  wins=" << wins << " expected=" << slots
                << " (each slot must have exactly one winner)" << std::endl;
      return 1;
    }
    for (uint32_t i = 0; i < slots; ++i) {
      uint64_t v = word64(shared, i);
      uint32_t k = (uint32_t)v - 1;
      if (k >= n || v != amo64_claim_value(k)) {
        std::cout << "  slot " << i << " holds 0x" << std::hex << v << std::dec
                  << ", expected a hart's claim" << std::endl;
        return 1;
      }
    }
    return 0;
  }
};
#endif

#if VX_CFG_XLEN == 32
// B1: a slow load into rd+1 then amoadd.d. Every returned old value must be
// decodable (per-hart bad count 0) and the counter must total base + n*step.
class Test_AMO64_WAR : public ITestCase {
public:
  Test_AMO64_WAR(TestSuite* s) : ITestCase(s, "amo64_war") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    set_word64(shared, 0, AMO64_ADD_BASE);
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* per_hart) override {
    int err = check_no_bad_olds(n, per_hart);
    err += check64("counter", word64(shared, 0),
                   AMO64_ADD_BASE + (uint64_t)n * iters * AMO64_ADD_STEP);
    return err;
  }
};

class Test_AMO64_WAR_LMEM : public ITestCase {
public:
  Test_AMO64_WAR_LMEM(TestSuite* s) : ITestCase(s, "amo64_war_lmem") {}
  void setup(uint32_t, uint32_t*, uint32_t*) override {}
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* per_hart) override {
    int err = check_no_bad_olds(n, per_hart);
    auto leaders = amo64_lmem_leaders(n, shared);
    for (uint32_t h : leaders) {
      uint32_t gs = (uint32_t)(word64(shared, h * 8 + 7) & 0xffffu);
      err += check64("lmem counter", word64(shared, h * 8),
                     AMO64_ADD_BASE + (uint64_t)gs * iters * AMO64_ADD_STEP);
    }
    return err;
  }
};

// B2: each thread stores 0 then atomically adds 1 to its own word; the store
// must order before the atomic, so the per-hart bad count must be 0.
class Test_AMO64_STORE_AMO : public ITestCase {
public:
  Test_AMO64_STORE_AMO(TestSuite* s) : ITestCase(s, "amo64_store_amo") {}
  void setup(uint32_t, uint32_t*, uint32_t*) override {}
  int verify(uint32_t n, uint32_t,
             const uint32_t*, const uint32_t* per_hart) override {
    return check_no_bad_olds(n, per_hart);
  }
};

class Test_AMO64_STORE_AMO_LMEM : public ITestCase {
public:
  Test_AMO64_STORE_AMO_LMEM(TestSuite* s) : ITestCase(s, "amo64_store_amo_lmem") {}
  void setup(uint32_t, uint32_t*, uint32_t*) override {}
  int verify(uint32_t n, uint32_t,
             const uint32_t*, const uint32_t* per_hart) override {
    return check_no_bad_olds(n, per_hart);
  }
};

// The zero pair as a discarded destination: each hart's word holds its last
// stored value plus the step, and every read-back matched.
class Test_AMO64_X0 : public ITestCase {
public:
  Test_AMO64_X0(TestSuite* s) : ITestCase(s, "amo64_x0") {}
  void setup(uint32_t, uint32_t*, uint32_t*) override {}
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* per_hart) override {
    int err = check_no_bad_olds(n, per_hart);
    for (uint32_t h = 0; h < n && iters != 0; ++h) {
      uint64_t got = (uint64_t)shared[h * 16] | ((uint64_t)shared[h * 16 + 1] << 32);
      uint64_t expected = amo64_x0_value(h, iters - 1) + AMO64_ADD_STEP;
      if (got != expected) {
        std::cout << "  hart " << h << " word=0x" << std::hex << got
                  << " expected=0x" << expected << std::dec << std::endl;
        ++err;
      }
    }
    return err;
  }
};
#endif
#endif

#if VX_CFG_XLEN == 32
// Refused .D encodings trap as illegal instructions and leave memory alone.
class Test_AMO64_ILLEGAL : public ITestCase {
public:
  Test_AMO64_ILLEGAL(TestSuite* s) : ITestCase(s, "amo64_illegal") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    shared[0] = 0x11111111u;
    shared[1] = 0x22222222u;
  }
  int verify(uint32_t n, uint32_t,
             const uint32_t* shared, const uint32_t* per_hart) override {
    const uint32_t c = VX_TRAP_CAUSE_ILLEGAL_INSTR;
    const uint32_t expected = c | (c << 8) | (c << 16);
    for (uint32_t h = 0; h < n; ++h) {
      if (per_hart[h] != expected) {
        std::cout << "  hart " << h << " trap causes=0x" << std::hex << per_hart[h]
                  << " expected=0x" << expected << std::dec << std::endl;
        return 1;
      }
    }
    if (shared[0] != 0x11111111u || shared[1] != 0x22222222u) {
      std::cout << "  a refused atomic changed memory" << std::endl;
      return 1;
    }
    return 0;
  }
};
#endif

inline TestSuite::TestSuite(vx_device_h device) : device_(device) {
  // Test_ATOMIC_CRITICAL needs to know VX_CFG_NUM_THREADS to compute the
  // expected count; query it from the device caps.
  uint64_t num_threads = 0;
  vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads);

  add_test(new Test_AMOADD(this));
  add_test(new Test_AMOOR(this));
  add_test(new Test_AMOAND(this));
  add_test(new Test_AMOXOR(this));
  add_test(new Test_AMOMAX(this));
  add_test(new Test_AMOMINU(this));
  add_test(new Test_AMOSWAP(this));
  add_test(new Test_LRSC_COUNTER(this));
  add_test(new Test_AMOADD_AQRL(this));
  add_test(new Test_LRSC_COUNTER_AQRL(this));
  add_test(new Test_ATOMIC_REDUCTION(this));
  add_test(new Test_ATOMIC_CRITICAL(this, (uint32_t)num_threads));
  add_test(new Test_SELF_CONSISTENCY(this));
  add_test(new Test_AMOADD_LMEM(this));
  add_test(new Test_LRSC_LMEM(this));
#if VX_CFG_EXT_ZACAS_ENABLED
  // Native compare-and-swap only exists when Zacas is built in. Without it a
  // compare-exchange lowers to an LR/SC retry loop, which is the construct
  // this instruction exists to replace and which does not survive contention.
  add_test(new Test_CAS_LADDER(this));
  add_test(new Test_CAS_LMEM(this));
  add_test(new Test_CAS_COUNTER(this));
  add_test(new Test_CAS_COUNTER_LMEM(this));
#endif
#if AMO64_TESTS
  add_test(new Test_AMO64_ARITH(this));
  add_test(new Test_AMO64_ARITH_LMEM(this));
  add_test(new Test_AMO64_BITS(this));
  add_test(new Test_AMO64_BITS_LMEM(this));
#if VX_CFG_EXT_ZACAS_ENABLED
  add_test(new Test_AMO64_CAS_COUNTER(this));
  add_test(new Test_AMO64_CAS_COUNTER_LMEM(this));
  add_test(new Test_AMO64_CAS_LADDER(this));
#endif
#if VX_CFG_XLEN == 32
  add_test(new Test_AMO64_WAR(this));
  add_test(new Test_AMO64_WAR_LMEM(this));
  add_test(new Test_AMO64_STORE_AMO(this));
  add_test(new Test_AMO64_STORE_AMO_LMEM(this));
  add_test(new Test_AMO64_X0(this));
#endif
#endif
#if VX_CFG_XLEN == 32
  add_test(new Test_AMO64_ILLEGAL(this));
#endif
}

inline TestSuite::~TestSuite() {
  for (auto* t : _tests) delete t;
}
