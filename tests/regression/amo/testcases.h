#pragma once

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <vortex.h>

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
//     of each warp participates (warp-lockstep workaround). Active
//     harts = num_cores * num_warps = num_harts / num_threads.
//     Final counter = active * iters; lock must end at 0.
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

inline TestSuite::TestSuite(vx_device_h device) : device_(device) {
  // Test_ATOMIC_CRITICAL needs to know NUM_THREADS to compute the
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
}

inline TestSuite::~TestSuite() {
  for (auto* t : _tests) delete t;
}
