// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// simobject.h: the SimX simulation kernel and its component API.
//
// A model is a set of units joined by ports. A unit reacts in one of two
// ways, never both:
//
//   one input            a handler port: `bool execute(T& pkt)` is called for
//                        the packet at the head of the port. Return true when
//                        you took it (the port pops it), false to leave it.
//   several inputs       `void on_tick()`: runs on every cycle, or, when the
//   or internal state    unit defines `bool busy()`, only on cycles where one
//                        of its inputs holds a packet or busy() is true.
//
// A unit that defines busy() sleeps when idle and is woken only when it has
// something to do: a packet becomes visible on one of its ports, an output it
// found full frees a slot, or another unit calls tick_wake(). A fixed latency
// is a property of the connection.
//
//   class Alu : public SimObject<Alu> {
//   public:
//     SimPortIn<instr_trace_t*>  input{this, 2, &Alu::execute};
//     SimPortOut<instr_trace_t*> output{this};
//     using SimObject::SimObject;                // (const SimContext&, name)
//     bool execute(instr_trace_t*& trace) {
//       if (output.full()) { return false; }
//       output.send(trace, latency(trace));
//       return true;
//     }
//   };
//   auto alu = Alu::Create("alu");
//   alu->output.bind(&next->input);
//
//   SimObject<U>   void on_tick(), bool busy(), void on_reset() (all
//                  optional; the hooks may be protected with
//                  `friend class SimObject<U>;`, or virtual in an intermediate
//                  base U); tick_wake(), name(); create with U::Create(name, ...)
//   SimPortIn<T>   a port of the unit that READS it: SimPortIn<T> in{this,
//                  capacity} read by on_tick(), or SimPortIn<T> in{this,
//                  capacity, &U::execute}. empty(), peek(), pop(), size(),
//                  set_registered_ready(true) for a port whose freed slot is
//                  reusable next cycle only, push(pkt, latency) to stage into
//                  the owner's own port, bypass_to(out[, conv]) to make the
//                  port a wire to out's reader, pop_callback(fn) for a credit
//                  return.
//   SimPortOut<T>  a port of the unit that WRITES it: bind(&in, latency),
//                  full(), send(pkt[, latency]), try_send(), size(), capacity()
//   SimPortInArray<T>, SimPortOutArray<T>
//                  port arrays; ready_mask() for arbitration
//
// Timing:
// - Within a cycle, units and handler ports run in creation order (a handler
//   port as it is constructed, a unit with on_tick() as Create() returns).
// - A packet sent at cycle c with latency d is visible at cycle c + d, in
//   send order among packets visible the same cycle. Latency 0 is visible at
//   once to units after the sender, next cycle to the others.
// - Capacity counts packets in flight and visible. A popped slot is reusable
//   the same cycle unless the port has a registered ready.
// - A handler port takes at most one packet per cycle.
//
// The functional kernel (-DSIMX_FUNCTIONAL) keeps the executor and drops the
// timing: latencies above one cycle become one, and ports never fill.

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "smallfunc.h"
#include "util.h"

// The platform's per-thread execution context is read on every send and
// cycles() query. initial-exec turns each access into one %fs-relative load
// instead of a __tls_get_addr call. It also puts libsimx's whole TLS segment
// in glibc's static block, whose spare room for dlopen'ed libraries is a few
// hundred bytes shared by all of them: keep thread_local state in libsimx to
// a few pointers.
#ifndef SIMX_TLS_IE
#define SIMX_TLS_IE __attribute__((tls_model("initial-exec")))
#endif

namespace vortex {

// A timed and a functional build must not link together: the functional
// kernel's symbols live in their own inline namespace.
#ifdef SIMX_FUNCTIONAL
inline namespace sim_functional {
inline constexpr bool kSimFunctional = true;
#else
inline constexpr bool kSimFunctional = false;
#endif

class SimPlatform;

class SimObjectBase;
class SimPortInBase;
class SimPortOutBase;
template <typename T> class SimPortIn;
template <typename T> class SimPortOut;

class SimContext {
private:
  SimContext() = default;
  friend class SimPlatform;
};

// An entry of its domain's tick scan: a unit with an on_tick(), or a handler
// port. It is awake while its bit in the domain's ready set is set.
class SimTask {
public:
  SimTask(const SimTask&) = delete;
  SimTask& operator=(const SimTask&) = delete;
  virtual ~SimTask() = default;

  // The execution domain the task was built in (see SimPlatform::DomainScope).
  uint32_t domain() const { return domain_; }

protected:
  inline SimTask();
  virtual void run() = 0;
  void sleep() { *word_ &= ~bit_; }

private:
  uint64_t* word_ = &inert_;   // its ready-set word, once in a scan
  uint64_t bit_ = 1;
  uint64_t inert_ = 0;
  void* scan_domain_ = nullptr;
  uint32_t scan_idx_ = ~0u;
  uint32_t domain_;
  friend class SimPlatform;
};

///////////////////////////////////////////////////////////////////////////////
// SimPlatform
///////////////////////////////////////////////////////////////////////////////

class SimPlatform {
public:
  __attribute__((always_inline)) static SimPlatform& instance() {
    static SimPlatform s_inst;
    return s_inst;
  }

  bool initialize() { return true; }
  void finalize() { this->cleanup(); }

  // In domain context this is the cycle being ticked: thread-local, so
  // workers advance without a global synchronization point. Host context
  // sees the committed global counter.
  uint64_t cycles() const {
    return (tl_exec_domain_ == HOST_DOMAIN) ? cycles_ : tl_cycles_;
  }

  // Counts resets: cycle numbers repeat across runs, so per-cycle state is
  // tagged with the run it belongs to.
  uint64_t reset_gen() const { return reset_gen_; }

  // The execution context of sends issued outside any tick (host calls).
  static constexpr uint32_t HOST_DOMAIN = 0xffffffffu;

  // No packet sent and not yet taken, and no cross-domain call pending: the
  // only valid quiescence test.
  bool idle() const {
    int64_t inflight = 0;
    for (auto& dom : domains_) {
      inflight += dom->inflight;
    }
    return inflight == 0 && this->cross_pending() == 0;
  }

  // Domain topology, used by the topology containers (Socket); units inherit
  // the scope in effect when they are created.
  uint32_t alloc_domain() { return num_domains_++; }
  uint32_t num_domains() const { return num_domains_; }
  uint32_t build_domain() const { return build_domain_; }
  bool build_launch_kept() const { return build_launch_kept_; }

  // True while reset_launch() runs: a unit resets its launch state but keeps
  // what only a device reset clears.
  bool launch_resetting() const { return launch_resetting_; }

  // Cap the worker-thread count (before the first tick).
  void mt_limit(uint32_t max_threads) { mt_max_threads_ = std::max(1u, max_threads); }

  // The worker-thread count (before the first tick); 0 or 1 runs serially.
  // Cycle results are identical for every value.
  void set_num_workers(uint32_t num_workers) { req_workers_ = std::max(1u, num_workers); }

  // Synchronization quantum (before the first tick): domains run this many
  // cycles between barriers, and a cross-domain call runs at the first quantum
  // boundary at or after its due cycle. Functional kernel only: the timed
  // kernel's cross-domain latencies are physical.
  void set_quantum(uint32_t cycles) {
    if constexpr (kSimFunctional) {
      __assert(!mt_init_done_, "the quantum is fixed once execution starts");
      quantum_ = std::max(1u, cycles);
    } else {
      __assert(cycles <= 1, "a synchronization quantum requires the functional kernel");
    }
  }

  // Units built under a LaunchKeepScope (the last-level cache and the memory
  // below it, when the command processor writes into that cache) keep their
  // state and in-flight work across reset_launch().
  class LaunchKeepScope {
  public:
    explicit LaunchKeepScope(bool keep) {
      auto& p = SimPlatform::instance();
      saved_ = p.build_launch_kept_;
      p.build_launch_kept_ = saved_ || keep;
    }
    ~LaunchKeepScope() { SimPlatform::instance().build_launch_kept_ = saved_; }
    LaunchKeepScope(const LaunchKeepScope&) = delete;
    LaunchKeepScope& operator=(const LaunchKeepScope&) = delete;
  private:
    bool saved_;
  };

  class DomainScope {
  public:
    explicit DomainScope(uint32_t domain) {
      auto& p = SimPlatform::instance();
      saved_ = p.build_domain_;
      p.build_domain_ = domain;
    }
    // The domain an existing unit lives in: a boundary stage owned by a
    // peer's domain rather than the current scope.
    explicit DomainScope(const SimTask* task) : DomainScope(task->domain()) {}
    ~DomainScope() { SimPlatform::instance().build_domain_ = saved_; }
    DomainScope(const DomainScope&) = delete;
    DomainScope& operator=(const DomainScope&) = delete;
  private:
    uint32_t saved_;
  };

  // True when the executing context must not touch state of `domain`
  // directly: inside another domain's tick, not host context.
  bool is_cross_exec(uint32_t domain) const {
    return tl_exec_domain_ != HOST_DOMAIN && domain != tl_exec_domain_;
  }

  template <typename U, typename... Args>
  std::shared_ptr<U> create_object(Args&&... args);

  void reset();
  void reset_launch();
  void tick();

  // Tick until stop() returns true; stop() runs in host context after every
  // cycle, as in `do { tick(); } while (!stop());`, which it equals cycle for
  // cycle. With workers, the check runs in the release step of the cycle-end
  // barrier, so a cycle costs one barrier instead of tick()'s two.
  template <typename Stop>
  void run_until(Stop&& stop);

  // Tick until stop(busy) returns true, for a loop that cannot end while any
  // of `voters` is busy(voter). After each cycle the thread ticking a voter
  // evaluates it, and stop runs with busy set when one was: it must then
  // return false and touch only uncore state, since other units may be
  // ticking. stop(false) sees exactly the state run_until(stop) would. With
  // workers, neighbor synchronization replaces the barrier (sync_loop).
  template <typename Stop, typename Obj, typename Busy>
  void run_until(Stop&& stop, const std::vector<Obj*>& voters, Busy&& busy);

  // Run `fn` in `domain`'s context at the start of its next cycle.
  template <typename F>
  void run_in_domain(uint32_t domain, F&& fn) { this->cross_call(domain, std::forward<F>(fn)); }

  void wake(SimTask* task) { *task->word_ |= task->bit_; }
  inline void wake_at(SimTask* task, uint64_t cycle);
  SimTask* running_task() const { return tl_exec_task_; }

private:
  // Per-domain state: scan list, ready set, timed wake-ups and counters are
  // private to one execution domain, so a domain can be ticked by its own
  // worker without touching another domain's structures.
  struct CrossEntry;
  struct Domain {
    std::vector<SimTask*> scan;   // creation order within the domain
    // Bit i is set while scan[i] is awake: a cycle visits only the awake
    // tasks, still in creation order. The words come in cache-line blocks,
    // each domain's written by its own worker only.
    struct alignas(64) ActiveBlock {
      uint64_t words[8] = {};
    };
    std::vector<ActiveBlock> active_blocks;
    uint32_t active_words = 0;
    uint64_t* active() { return active_blocks.data()->words; }
    // Packets in flight toward, or held in, this domain's ports.
    alignas(64) int64_t inflight = 0;
    // The cycle this domain is ticking; the next one between cycles. Ports
    // compare their head's ready cycle against it.
    uint64_t now = 0;
    // Timed wake-ups (wake_at): ready-set masks for the next 64 cycles, and
    // a heap beyond.
    static constexpr uint32_t kWakeSlots = 64;
    std::vector<uint64_t> wake_slots[kWakeSlots];
    uint64_t wake_armed = 0;
    std::vector<std::pair<uint64_t, uint32_t>> wake_far;  // min-heap on cycle
    std::vector<CrossEntry*> cross_order;  // drain scratch (quantum > 1)
    uint32_t id = 0;
  };

  Domain& domain_at(uint32_t d) {
    if (__builtin_expect(d < domains_.size(), 1)) {
      return *domains_[d];
    }
    return this->domain_grow(d);
  }

  __attribute__((noinline)) Domain& domain_grow(uint32_t d) {
    __assert(!workers_started_, "domain created during parallel execution");
    while (domains_.size() <= d) {
      domains_.push_back(std::make_unique<Domain>());
      domains_.back()->id = uint32_t(domains_.size() - 1);
    }
    return *domains_[d];
  }

  // Elaboration only: add a task to its domain's scan list, awake.
  void add_scan(SimTask* task) {
    auto& dom = this->domain_at(task->domain_);
    uint32_t idx = uint32_t(dom.scan.size());
    dom.scan.push_back(task);
    if ((idx & 63) == 0) {
      if ((dom.active_words & 7) == 0) {
        auto* before = dom.active_blocks.data();
        dom.active_blocks.emplace_back();
        if (dom.active_blocks.data() != before) {
          for (uint32_t i = 0; i < idx; ++i) {
            dom.scan[i]->word_ = &dom.active()[i >> 6];
          }
        }
      }
      ++dom.active_words;
      for (auto& slot : dom.wake_slots) {
        slot.resize(dom.active_words, 0);
      }
    }
    task->word_ = &dom.active()[idx >> 6];
    task->bit_ = uint64_t(1) << (idx & 63);
    task->scan_idx_ = idx;
    task->scan_domain_ = &dom;
    *task->word_ |= task->bit_;
  }

  // Wake every task (a reset re-arms them all).
  void wake_all() {
    for (auto& dom : domains_) {
      uint32_t n = uint32_t(dom->scan.size());
      for (uint32_t w = 0; w < dom->active_words; ++w) {
        uint32_t bits = std::min(64u, n - w * 64);
        dom->active()[w] = (bits == 64) ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1);
      }
    }
  }

  SimPlatform() = default;
  ~SimPlatform() { this->cleanup(); }

  inline void cleanup();
  inline void tick_domain(Domain& dom, bool drain = true);
  inline void validate_topology();

  // One cycle (functional: one quantum) of a domain, then its clock moves
  // to the next cycle for the host and for cross-domain arrivals.
  void run_domain(Domain& dom, uint64_t base) {
    for (uint64_t q = 0; q < quantum_; ++q) {
      tl_cycles_ = base + q;
      this->tick_domain(dom, q == 0);
      dom.now = base + q + 1;
    }
  }

  std::vector<std::shared_ptr<SimObjectBase>> objects_;
  std::vector<SimObjectBase*> resets_;   // units with an on_reset(), in creation order
  std::vector<SimPortOutBase*> outs_;      // every output port, for the topology
  std::vector<std::unique_ptr<Domain>> domains_;

  uint64_t cycles_ = 0;
  uint64_t reset_gen_ = 0;
  uint64_t quantum_ = 1;

  uint32_t num_domains_ = 1;
  uint32_t build_domain_ = 0;
  bool build_launch_kept_ = false;
  bool launch_resetting_ = false;
  bool topo_validated_ = false;
  static inline thread_local uint32_t tl_exec_domain_ SIMX_TLS_IE = HOST_DOMAIN;
  static inline thread_local SimTask* tl_exec_task_ SIMX_TLS_IE = nullptr;
  static inline thread_local uint64_t tl_cycles_ SIMX_TLS_IE = 0;

  // ── Cross-domain calls ───────────────────────────────────────────────
  // A call runs in the target domain at the start of its next cycle (with a
  // quantum, at the first quantum boundary at or after it), in canonical
  // order: due cycle, then source domain (host last), then append order.
  // Each (target, source) pair owns a lane, so a source appends without
  // locks and the target drains alone. A lane is triple-buffered by epoch (a
  // quantum): during epoch e a source writes epoch e, while the target
  // drains epoch e-1 after the barrier that closed it.
  using CrossFn = SmallFunction<void(), 160>;
  static constexpr uint32_t kCrossEpochs = 3;
  static constexpr uint32_t kHostLane = 0;   // lane 1 + d belongs to domain d

  struct CrossEntry {
    uint64_t due;
    CrossFn fn;
  };
  struct alignas(64) CrossLane {
    std::vector<CrossEntry> buf[kCrossEpochs];
  };
  std::vector<std::vector<std::unique_ptr<CrossLane>>> mail_;   // [target][lane]

  uint64_t cross_epoch(uint64_t cycle) const { return cycle / quantum_; }

  // Undelivered calls (host context only: workers parked).
  uint64_t cross_pending() const {
    uint64_t n = 0;
    for (auto& lanes : mail_) {
      for (auto& lane : lanes) {
        if (lane) {
          for (auto& b : lane->buf) {
            n += b.size();
          }
        }
      }
    }
    return n;
  }

  // Host-context calls must come from one thread at a time: they share the
  // host lane.
  void cross_call(uint32_t target, CrossFn&& fn) {
    assert(target < num_domains_);
    uint32_t src = tl_exec_domain_;
    uint64_t now = this->cycles();
    if (sync_active_ && src != HOST_DOMAIN && src != target && !sync_edge_[target * num_domains_ + src]) {
      this->sync_unknown_edge(src, target);
    }
    auto& lane = this->cross_lane(target, (src == HOST_DOMAIN) ? kHostLane : (src + 1));
    lane.buf[this->cross_epoch(now) % kCrossEpochs].push_back({now + 1, std::move(fn)});
  }

  CrossLane& cross_lane(uint32_t target, uint32_t lane) {
    if (__builtin_expect(target < mail_.size(), 1)) {
      auto& lanes = mail_[target];
      if (__builtin_expect(lane < lanes.size() && lanes[lane] != nullptr, 1)) {
        return *lanes[lane];
      }
    }
    return this->cross_lane_grow(target, lane);
  }

  // Elaboration or serial execution only: parallel workers find every lane
  // materialized by init_mt.
  __attribute__((noinline)) CrossLane& cross_lane_grow(uint32_t target, uint32_t lane) {
    __assert(!workers_started_, "cross-domain lane created during parallel execution");
    if (mail_.size() <= target) {
      mail_.resize(target + 1);
    }
    auto& lanes = mail_[target];
    if (lanes.size() <= lane) {
      lanes.resize(lane + 1);
    }
    if (!lanes[lane]) {
      lanes[lane] = std::make_unique<CrossLane>();
    }
    return *lanes[lane];
  }

  void clear_cross_lanes() {
    for (auto& lanes : mail_) {
      for (auto& lane : lanes) {
        if (lane) {
          for (auto& b : lane->buf) {
            b.clear();
          }
        }
      }
    }
  }

  // Run the calls due this epoch for domain d: the previous epoch's buffer
  // of each lane, in canonical order. A call may create a lane in serial
  // execution, which can reallocate mail_, so lanes are indexed afresh.
  void drain_cross_calls(uint32_t d) {
    if (d >= mail_.size()) {
      return;
    }
    uint32_t nlanes = uint32_t(mail_[d].size());
    uint32_t idx = uint32_t((this->cross_epoch(this->cycles()) + kCrossEpochs - 1) % kCrossEpochs);
    bool any = false;
    for (auto& lane : mail_[d]) {
      any |= (lane && !lane->buf[idx].empty());
    }
    if (!any) {
      return;
    }
    if (quantum_ == 1) {
      // every entry has the same due cycle: source order, host last
      for (uint32_t l = 1; l <= nlanes; ++l) {
        auto* lane = mail_[d][l % nlanes].get();
        if (lane) {
          for (auto& e : lane->buf[idx]) {
            e.fn();
          }
          lane->buf[idx].clear();
        }
      }
      return;
    }
    // Workers drain their domains concurrently: each domain has its own scratch.
    auto& order = this->domain_at(d).cross_order;
    order.clear();
    for (uint32_t l = 1; l <= nlanes; ++l) {
      if (auto* lane = mail_[d][l % nlanes].get()) {
        for (auto& e : lane->buf[idx]) {
          order.push_back(&e);
        }
      }
    }
    std::stable_sort(order.begin(), order.end(),
                     [](const CrossEntry* a, const CrossEntry* b) { return a->due < b->due; });
    for (auto* e : order) {
      e->fn();
    }
    for (auto& lane : mail_[d]) {
      if (lane) {
        lane->buf[idx].clear();
      }
    }
  }

  // ── Multi-threaded executor ──────────────────────────────────────────
  // One persistent worker per extra thread; domains are assigned round-robin
  // (the coordinator, thread 0, keeps the uncore domain). A cycle is: each
  // thread ticks its domains, then one barrier. All per-domain state is
  // touched only by its owner; the only shared structures are the mailbox
  // lanes (one writer and one reader each, kept apart by epoch) and the
  // barrier. Inside run_until() the cycle-end barrier also starts the next
  // cycle: its release step commits the cycle and runs the stop check.
  uint32_t mt_threads_ = 1;
  uint32_t req_workers_ = 1;
  uint32_t mt_max_threads_ = std::numeric_limits<uint32_t>::max();
  bool mt_init_done_ = false;
  bool workers_started_ = false;
  std::vector<std::thread> workers_;
  std::vector<std::vector<Domain*>> stripe_domains_;
  std::atomic<uint64_t> barrier_epoch_{0};
  std::atomic<bool> workers_exit_{false};

  // Each worker announces its arrival on its own cache line.
  struct alignas(64) ArriveFlag {
    std::atomic<uint64_t> epoch{0};
  };
  std::unique_ptr<ArriveFlag[]> arrive_;

  bool (*run_stop_)(void*) = nullptr;
  void* run_stop_arg_ = nullptr;
  bool run_active_ = false;

  // ── Neighbor synchronization (run_until with voters) ─────────────────
  // Domain d publishes the cycles it has completed and its vote, and starts
  // the quantum at cycle c once every neighbor has completed c, so neighbors
  // stay within one quantum and the triple-buffered mailboxes stay valid.
  // Thread 0 owns the uncore domain, which neighbors every domain, and
  // decides the stop check for the quantum ending at c when it starts the
  // next one. A thread with a busy domain may run one quantum ahead, which
  // rules out a stop at c; a thread whose domains all voted idle waits for
  // the decision, so an all-idle check runs with every other thread parked.
  struct alignas(64) SyncProgress {
    std::atomic<uint64_t> word{0};   // completed cycles << 1 | busy vote
  };
  std::unique_ptr<SyncProgress[]> sync_progress_;
  std::vector<std::vector<uint32_t>> sync_deps_;   // neighbors per domain
  std::vector<uint8_t> sync_edge_;                 // [target * num_domains_ + source]
  alignas(64) std::atomic<uint64_t> sync_decided_{0};
  std::atomic<bool> sync_stop_{false};
  bool sync_active_ = false;
  uint64_t sync_start_ = 0;
  bool (*sync_stop_fn_)(void*, bool) = nullptr;
  bool (*sync_vote_fn_)(void*, uint32_t) = nullptr;
  void* sync_stop_arg_ = nullptr;
  void* sync_vote_arg_ = nullptr;

  // Neighbors: domains joined by a port, and the uncore with every domain.
  // A cross-domain call along any other pair aborts the run (cross_call).
  inline void sync_init_graph();

  __attribute__((noinline, noreturn)) void sync_unknown_edge(uint32_t src, uint32_t target) {
    std::cerr << "error: cross-domain call from domain " << src << " to domain " << target
              << ", which share no port; neighbor synchronization cannot order it" << std::endl;
    std::abort();
  }

  // Thread w's part of a synchronized run_until().
  void sync_loop(uint32_t w) {
    const uint64_t step = quantum_;
    for (uint64_t c = sync_start_;; c += step) {
      bool any_busy = false;
      for (auto* dom : stripe_domains_[w]) {
        if (dom->id == 0 && c > sync_start_) {
          bool busy = (sync_progress_[0].word.load(std::memory_order_relaxed) & 1) != 0;
          for (uint32_t d = 1; d < num_domains_; ++d) {
            uint64_t word;
            spin_until([&]() {
              word = sync_progress_[d].word.load(std::memory_order_acquire);
              return (word >> 1) >= c;
            });
            busy |= ((word >> 1) > c) || (word & 1);   // only a busy thread runs ahead
          }
          cycles_ = c;
          tl_exec_domain_ = HOST_DOMAIN;
          tl_exec_task_ = nullptr;
          if (sync_stop_fn_(sync_stop_arg_, busy)) {
            __assert(!busy, "run_until: stop() returned true on a busy cycle");
            sync_stop_.store(true, std::memory_order_release);
            return;
          }
          sync_decided_.store(c, std::memory_order_release);
        }
        for (auto n : sync_deps_[dom->id]) {
          spin_until([&]() { return (sync_progress_[n].word.load(std::memory_order_acquire) >> 1) >= c; });
        }
        this->run_domain(*dom, c);
        bool vote = sync_vote_fn_(sync_vote_arg_, dom->id);
        any_busy |= vote;
        sync_progress_[dom->id].word.store(((c + step) << 1) | (vote ? 1 : 0), std::memory_order_release);
      }
      tl_exec_domain_ = HOST_DOMAIN;
      tl_exec_task_ = nullptr;
      if (w != 0 && !any_busy) {
        spin_until([&]() {
          return sync_stop_.load(std::memory_order_acquire)
              || sync_decided_.load(std::memory_order_acquire) >= c + step;
        });
        if (sync_stop_.load(std::memory_order_acquire)) {
          return;
        }
      }
    }
  }

  // Spin briefly (peers usually arrive within microseconds), then yield so a
  // waiting thread does not starve a descheduled peer.
  template <typename Cond>
  static void spin_until(Cond&& cond) {
    uint32_t spins = 0;
    while (!cond()) {
      if (++spins < 4096) {
      #if defined(__x86_64__)
        __builtin_ia32_pause();
      #endif
      } else {
        std::this_thread::yield();
      }
    }
  }

  // Thread `self` arrives (0 is the coordinator). The coordinator waits for
  // every worker, runs on_release and publishes the new epoch, so the
  // release step's writes are visible on the other side of the barrier.
  template <typename Fn>
  void barrier_wait(uint32_t self, Fn&& on_release) {
    uint64_t epoch = barrier_epoch_.load(std::memory_order_acquire);
    if (self == 0) {
      for (uint32_t i = 1; i < mt_threads_; ++i) {
        spin_until([&]() { return arrive_[i].epoch.load(std::memory_order_acquire) == epoch + 1; });
      }
      on_release();
      barrier_epoch_.store(epoch + 1, std::memory_order_release);
    } else {
      arrive_[self].epoch.store(epoch + 1, std::memory_order_release);
      spin_until([&]() { return barrier_epoch_.load(std::memory_order_acquire) != epoch; });
    }
  }

  void barrier_wait(uint32_t self) { this->barrier_wait(self, []() {}); }

  void run_stripe(uint32_t w) {
    for (auto* dom : stripe_domains_[w]) {
      this->run_domain(*dom, cycles_);
    }
    tl_exec_domain_ = HOST_DOMAIN;
    tl_exec_task_ = nullptr;
  }

  // Release step of the cycle-end barrier: commit the cycle and, inside a
  // run_until() loop, evaluate the stop check.
  void cycle_end() {
    cycles_ += quantum_;
    if (run_active_ && run_stop_(run_stop_arg_)) {
      run_active_ = false;
    }
  }

  void init_mt(uint32_t num_workers) {
    mt_init_done_ = true;
    uint32_t count = std::min({std::max(1u, num_workers), num_domains_, mt_max_threads_});
    if (count < 2) {
      return;
    }
    mt_threads_ = count;
    // Materialize every domain and lane up front: workers never resize them.
    (void)this->domain_at(num_domains_ - 1);
    for (uint32_t t = 0; t < num_domains_; ++t) {
      for (uint32_t l = 0; l <= num_domains_; ++l) {
        (void)this->cross_lane_grow(t, l);
      }
    }
    // Round-robin, with the last partial round placed from the highest
    // thread down: thread 0 owns the uncore domain and runs the release
    // steps, so it takes no extra domain.
    stripe_domains_.assign(count, {});
    uint32_t full_rounds = (num_domains_ / count) * count;
    for (uint32_t d = 0; d < num_domains_; ++d) {
      uint32_t t = (d < full_rounds) ? (d % count) : (count - 1 - (d - full_rounds));
      stripe_domains_[t].push_back(domains_[d].get());
    }
    arrive_.reset(new ArriveFlag[count]);
    barrier_epoch_.store(0, std::memory_order_relaxed);
    for (uint32_t w = 1; w < count; ++w) {
      workers_.emplace_back([this, w]() {
        while (true) {
          this->barrier_wait(w);   // parked between runs
          if (workers_exit_.load(std::memory_order_acquire)) {
            break;
          }
          if (sync_active_) {
            this->sync_loop(w);
            this->barrier_wait(w);   // join
            continue;
          }
          // One cycle for tick(); cycles until the stop check fires for
          // run_until(). run_active_ is written only in a release step.
          do {
            this->run_stripe(w);
            this->barrier_wait(w, [this]() { this->cycle_end(); });
          } while (run_active_);
        }
      });
    }
    workers_started_ = true;
  }

  void stop_workers() {
    if (!workers_started_) {
      return;
    }
    workers_exit_.store(true, std::memory_order_release);
    this->barrier_wait(0);
    for (auto& t : workers_) {
      t.join();
    }
    workers_.clear();
    arrive_.reset();
    stripe_domains_.clear();
    workers_started_ = false;
    workers_exit_.store(false, std::memory_order_relaxed);
    mt_threads_ = 1;
  }

  friend class SimTask;
  friend class SimPortInBase;
  friend class SimPortOutBase;
  template <typename T> friend class SimPortIn;
  template <typename T> friend class SimEventLink;
};

inline SimTask::SimTask() : domain_(SimPlatform::instance().build_domain()) {}

__attribute__((always_inline)) inline void SimPlatform::wake_at(SimTask* task, uint64_t cycle) {
  __assert(task->scan_domain_ != nullptr, "a timed wake-up needs a task in a scan");
  auto& dom = *static_cast<Domain*>(task->scan_domain_);
  if (cycle <= dom.now) {
    *task->word_ |= task->bit_;
    return;
  }
  uint32_t idx = task->scan_idx_;
  if (cycle - dom.now < Domain::kWakeSlots) {
    uint32_t slot = uint32_t(cycle & (Domain::kWakeSlots - 1));
    dom.wake_slots[slot][idx >> 6] |= uint64_t(1) << (idx & 63);
    dom.wake_armed |= uint64_t(1) << slot;
  } else {
    dom.wake_far.emplace_back(cycle, idx);
    std::push_heap(dom.wake_far.begin(), dom.wake_far.end(), std::greater<>());
  }
}

///////////////////////////////////////////////////////////////////////////////
// SimEventLink
///////////////////////////////////////////////////////////////////////////////

// One-way, typed event link for control strobes (doorbells, barrier arrive
// and resume). The receiver binds a member-function handler to its end; the
// sender's end binds to the receiver's (fan-in allowed). Delivery cannot be
// refused and takes a registered delay of at least one cycle: the handler
// runs in the receiver's domain at the start of cycle now + delay, ahead of
// its tick scan, in canonical order with other cross-domain calls. A link
// has no queue and no back-pressure; it is the way to act on a unit in
// another execution domain.
template <typename Msg>
class SimEventLink {
public:
  static_assert(std::is_copy_constructible_v<Msg>, "Message must be copy constructible");

  SimEventLink(const SimTask* owner) : domain_(owner->domain()) {}

  // Copying is only for container fill at construction: bindings do not
  // carry over.
  SimEventLink(const SimEventLink& other) : domain_(other.domain_) {}

  SimEventLink& operator=(const SimEventLink& other) {
    if (this != &other) {
      domain_ = other.domain_;
      sink_ = nullptr;
      handler_ = {};
    }
    return *this;
  }

  // Sender end: wire to the receiving end.
  void bind(SimEventLink<Msg>* sink) {
    __assert(sink != nullptr, "bind target is null");
    __assert(sink_ == nullptr && !handler_, "link end already bound");
    sink_ = sink;
  }

  // Receiver end: the member-function handler invoked on delivery.
  template <typename Impl>
  void bind(Impl* obj, void (Impl::*handler)(const Msg&)) {
    __assert(obj != nullptr, "handler object is null");
    __assert(sink_ == nullptr && !handler_, "link end already bound");
    handler_ = [obj, handler](const Msg& msg) { (obj->*handler)(msg); };
  }

  void send(const Msg& msg, uint64_t delay = 1) {
    __assert(delay >= 1, "event link delivery requires a registered delay");
    auto* ep = this;
    while (ep->sink_ != nullptr) {
      ep = ep->sink_;
    }
    __assert(!!ep->handler_, "event link has no bound handler");
    SimPlatform::instance().cross_call(ep->domain_, [ep, msg, delay]() {
      if (delay > 1) {
        ep->send(msg, delay - 1);   // re-issued from the receiver's own context
      } else {
        ep->handler_(msg);
      }
    });
  }

private:
  uint32_t domain_;
  SimEventLink<Msg>* sink_ = nullptr;
  SmallFunction<void(const Msg&), 48> handler_;
};

///////////////////////////////////////////////////////////////////////////////
// Units and ports
///////////////////////////////////////////////////////////////////////////////

namespace detail {

// Set when an output refused during the running handler.
inline thread_local bool tl_blocked SIMX_TLS_IE = false;

// A port's packets: each slot holds a packet and the cycle it becomes
// visible, in ready order (equal cycles in send order). A send writes its
// packet here at once. The functional kernel grows the ring on demand.
template <typename T>
class DelayRing {
public:
  explicit DelayRing(uint32_t capacity)
    : capacity_(capacity)
    , slots_(new Slot[capacity]) {
    __assert(capacity > 0, "a port needs a capacity");
  }
  DelayRing(const DelayRing&) = delete;
  DelayRing& operator=(const DelayRing&) = delete;
  ~DelayRing() {
    while (size_ != 0) {
      this->pop();
    }
  }

  uint32_t size() const { return size_; }
  uint32_t capacity() const { return capacity_; }

  // Ready cycle of the oldest packet; all ones when empty.
  uint64_t head_ready() const { return head_ready_; }
  const uint64_t* head_ready_ptr() const { return &head_ready_; }

  T& front() { return *slots_[head_].pkt(); }
  const T& front() const { return *slots_[head_].pkt(); }

  void pop() {
    slots_[head_].pkt()->~T();
    head_ = this->wrap(head_ + 1);
    --size_;
    head_ready_ = (size_ != 0) ? slots_[head_].ready : ~uint64_t(0);
  }

  template <typename P>
  void insert(P&& pkt, uint64_t ready) {
    if (size_ == capacity_) {
      __assert(kSimFunctional, "port overflow");
      this->grow();
    }
    uint32_t i = size_;
    uint32_t pos = this->wrap(head_ + i);
    while (i > 0) {
      uint32_t prev = this->wrap(head_ + i - 1);
      if (slots_[prev].ready <= ready) {
        break;
      }
      new (slots_[pos].buf) T(std::move(*slots_[prev].pkt()));
      slots_[pos].ready = slots_[prev].ready;
      slots_[prev].pkt()->~T();
      pos = prev;
      --i;
    }
    new (slots_[pos].buf) T(std::forward<P>(pkt));
    slots_[pos].ready = ready;
    ++size_;
    if (i == 0) {
      head_ready_ = ready;
    }
  }

private:
  struct Slot {
    uint64_t ready;
    alignas(T) unsigned char buf[sizeof(T)];
    T* pkt() { return std::launder(reinterpret_cast<T*>(buf)); }
    const T* pkt() const { return std::launder(reinterpret_cast<const T*>(buf)); }
  };

  uint32_t wrap(uint32_t i) const { return (i >= capacity_) ? i - capacity_ : i; }

  // Functional kernel: no back-pressure, so a port doubles when it fills; a
  // feedback loop that outruns its reader fails at the high-water mark.
  __attribute__((noinline)) void grow() {
    __assert(capacity_ < (1u << 20), "functional port high-water exceeded");
    std::unique_ptr<Slot[]> slots(new Slot[capacity_ * 2]);
    for (uint32_t i = 0; i < size_; ++i) {
      auto& src = slots_[this->wrap(head_ + i)];
      new (slots[i].buf) T(std::move(*src.pkt()));
      slots[i].ready = src.ready;
      src.pkt()->~T();
    }
    slots_ = std::move(slots);
    capacity_ *= 2;
    head_ = 0;
  }

  uint32_t capacity_;
  std::unique_ptr<Slot[]> slots_;
  uint32_t head_ = 0;
  uint32_t size_ = 0;
  uint64_t head_ready_ = ~uint64_t(0);
};

}  // namespace detail

/// The non-template part of every unit: its scan task and its ports.
class SimObjectBase : public SimTask {
public:
  const std::string& name() const { return name_; }

  /// Run the unit again from the earliest legal point: its on_tick(), or its
  /// handler ports. For state another unit changed by a direct call.
  inline void tick_wake();

protected:
  SimObjectBase(const std::string& name, bool ticks)
    : name_(name)
    , ticks_(ticks)
    , launch_kept_(SimPlatform::instance().build_launch_kept()) {}

private:
  virtual void do_reset() = 0;
  inline bool inputs_ready() const;

  std::string name_;
  bool ticks_;
  bool launch_kept_;
  std::vector<SimPortInBase*> plain_;
  std::vector<SimPortInBase*> handlers_;

  friend class vortex::SimPlatform;
  template <typename U> friend class SimObject;
  friend class SimPortInBase;
  template <typename T> friend class SimPortIn;
};

/// The base of every model. `using SimObject::SimObject;` inherits the
/// (const SimContext&, name) constructor; create units with U::Create(...).
/// The hooks are found on U, so U may keep them protected by declaring
/// `friend class SimObject<U>;`, or declare them virtual for its subclasses.
template <typename U>
class SimObject : public SimObjectBase {
  template <typename V>
  static auto detect_tick(int) -> decltype(std::declval<V&>().on_tick(), std::true_type{});
  template <typename V>
  static std::false_type detect_tick(...);
  template <typename V>
  static auto detect_reset(int) -> decltype(std::declval<V&>().on_reset(), std::true_type{});
  template <typename V>
  static std::false_type detect_reset(...);
  template <typename V>
  static auto detect_busy(int) -> decltype(bool(std::declval<V&>().busy()), std::true_type{});
  template <typename V>
  static std::false_type detect_busy(...);

public:
  using Ptr = std::shared_ptr<U>;
  static constexpr bool kTicks = decltype(detect_tick<U>(0))::value;
  static constexpr bool kResets = decltype(detect_reset<U>(0))::value;
  static constexpr bool kSleeps = decltype(detect_busy<U>(0))::value;

  template <typename... Args>
  static Ptr Create(Args&&... args) {
    return SimPlatform::instance().create_object<U>(std::forward<Args>(args)...);
  }

  SimObject(const SimContext&, const std::string& name) : SimObjectBase(name, kTicks) {}

private:
  void run() override {
    if constexpr (kTicks) {
      static_cast<U*>(this)->on_tick();
      // Without busy(), a unit is always busy: it ticks every cycle.
      if constexpr (!kSleeps) {
        return;
      } else if (this->inputs_ready() || static_cast<U*>(this)->busy()) {
        return;
      }
    }
    this->sleep();
  }

  void do_reset() override {
    if constexpr (kResets) {
      static_cast<U*>(this)->on_reset();
    }
  }
};

/// The non-template part of an input port.
class SimPortInBase {
public:
  SimPortInBase(const SimPortInBase&) = delete;
  SimPortInBase& operator=(const SimPortInBase&) = delete;
  virtual ~SimPortInBase() = default;

  /// A wire through the unit (see SimPortIn::bypass_to): the unit never reads it.
  bool bypassed() const { return bypassed_; }

  /// Packets in flight and visible, and the port's room.
  virtual uint32_t size() const = 0;
  virtual uint32_t capacity() const = 0;

  /// Called on every pop, in the reader's context: a credit return.
  template <typename F>
  void pop_callback(F&& fn) { pop_cb_.reset(new PopCallback(std::forward<F>(fn))); }

protected:
  using ServeFn = bool (*)(SimPortInBase*);   // true: run again next cycle

  // The scan entry of a handler port, woken by arrivals on its port.
  struct Entry final : SimTask {
    SimPortInBase* in = nullptr;
    ServeFn serve = nullptr;
    void run() override {
      detail::tl_blocked = false;
      if (!serve(in)) {
        this->sleep();
      }
    }
  };

  SimPortInBase(SimObjectBase* owner, ServeFn serve) : owner_(owner) {
    if (serve != nullptr) {
      __assert(!owner->ticks_, "a unit with on_tick() reads its ports itself: SimPortIn<T>{this, capacity}");
      entry_.reset(new Entry());
      entry_->in = this;
      entry_->serve = serve;
      SimPlatform::instance().add_scan(entry_.get());
      owner->handlers_.push_back(this);
      reader_ = entry_.get();
    } else {
      __assert(owner->ticks_, "a port without a handler is read by the unit's on_tick()");
      owner->plain_.push_back(this);
      reader_ = owner;
    }
  }

  void unlist() {
    auto& plain = owner_->plain_;
    plain.erase(std::remove(plain.begin(), plain.end(), this), plain.end());
    bypassed_ = true;
  }

  virtual bool has_room() const = 0;

  // A packet is visible: a plain load and compare, no virtual call, since
  // the owner asks this of every port after each tick.
  bool ready_now() const { return *head_ready_ <= *clock_; }

  virtual void wait_for_room(SimTask* waiter) = 0;

  // One cache line, read on every tick of the owner; the handler's scan entry
  // and the pop callback are allocated only by the ports that use them.
  using PopCallback = SmallFunction<void(), 48>;
  SimObjectBase* owner_;
  SimTask* reader_;   // woken when a packet becomes visible
  const uint64_t* head_ready_ = nullptr;   // the ring's head ready cycle
  const uint64_t* clock_ = nullptr;        // the owner domain's cycle
  std::unique_ptr<Entry> entry_;           // handler ports only
  std::unique_ptr<PopCallback> pop_cb_;
  bool bypassed_ = false;

  friend class SimObjectBase;
  friend class vortex::SimPlatform;
  template <typename T> friend class SimPortOut;
};

/// An output port, for the topology walk and the boundary stages.
class SimPortOutBase {
public:
  SimPortOutBase(const SimPortOutBase&) = delete;
  SimPortOutBase& operator=(const SimPortOutBase&) = delete;
  virtual ~SimPortOutBase() = default;

  /// The port this output's packets land in, through any wires; null while
  /// unbound.
  virtual SimPortInBase* endpoint() = 0;

  /// This output crosses an execution-domain boundary through a registered,
  /// credit-gated stage (RegSlice): parallel execution may cut here.
  void mark_boundary() { boundary_ = true; }

protected:
  explicit SimPortOutBase(SimObjectBase* owner) : owner_(owner) {
    SimPlatform::instance().outs_.push_back(this);
  }

  SimObjectBase* owner_;
  bool boundary_ = false;

  friend class vortex::SimPlatform;
};

/// An input port, owned by the unit that reads it.
template <typename T>
class SimPortIn final : public SimPortInBase {
public:
  /// A port read by the owner's on_tick().
  SimPortIn(SimObjectBase* owner, uint32_t capacity = 2)
    : SimPortInBase(owner, nullptr)
    , ring_(capacity) {
    this->bind_domain();
  }

  /// A handler port: `handler(pkt)` returns true when it took the packet.
  template <typename U>
  SimPortIn(U* owner, uint32_t capacity, bool (U::*handler)(T&))
    : SimPortInBase(owner, &SimPortIn::serve)
    , ring_(capacity)
    , handler_(static_cast<Handler>(handler)) {
    this->bind_domain();
  }

  /// Port `index` of a handler array: `handler(index, pkt)`.
  template <typename U>
  SimPortIn(U* owner, uint32_t capacity, bool (U::*handler)(uint32_t, T&), uint32_t index)
    : SimPortInBase(owner, &SimPortIn::serve_indexed)
    , ring_(capacity)
    , indexed_(static_cast<IndexedHandler>(handler))
    , index_(index) {
    this->bind_domain();
  }

  /// No packet visible this cycle.
  bool empty() const { return ring_.head_ready() > *clock_; }

  const T& peek() const {
    __assert(!this->empty(), "port is empty");
    return ring_.front();
  }

  /// Take the oldest visible packet.
  T pop() {
    __assert(!this->empty(), "port is empty");
    T pkt(std::move(ring_.front()));
    this->remove();
    return pkt;
  }

  /// The cycle the oldest packet became, or becomes, visible.
  uint64_t head_cycle() const { return ring_.head_ready(); }

  /// A bypassed port holds nothing itself: its packets are in flight to, or
  /// visible at, the port its wire ends in, so it reports that port's size.
  uint32_t size() const override {
    if (__builtin_expect(this->bypassed(), 0)) {
      return this->wire_size();
    }
    return ring_.size();
  }

  // Nominal in the functional kernel: large enough never to gate, small
  // enough for a credit counter.
  uint32_t capacity() const override {
    return kSimFunctional ? (1u << 30) : ring_.capacity();
  }

  /// The owner stages a packet into its own port, visible `latency` cycles
  /// on: an internal pipeline register. Requires !full().
  void push(T pkt, uint32_t latency) { this->arrive(std::move(pkt), latency); }
  bool full() const { return !this->has_room(); }

  /// A registered ready: a slot freed by a pop is offered to producers from
  /// the next cycle only. Set before simulation starts.
  void set_registered_ready(bool enable) { reg_ready_ = enable; }

  /// Make this port a wire to whatever `out` is bound to: producers bound
  /// here reach that reader directly, with no added latency.
  void bypass_to(SimPortOut<T>& out) {
    fwd_ = &out;
    this->unlist();
  }

  /// As bypass_to, converting each packet on the way: `conv(T) -> V`.
  template <typename V, typename Conv>
  void bypass_to(SimPortOut<V>& out, Conv&& conv) {
    conv_.reset(new Converted{&out, [&out, c = std::forward<Conv>(conv)](T&& pkt, uint64_t latency) {
      out.send(c(std::move(pkt)), latency);
    }});
    this->unlist();
  }

private:
  template <typename V> friend class SimPortOut;
  using Handler = bool (SimObjectBase::*)(T&);
  using IndexedHandler = bool (SimObjectBase::*)(uint32_t, T&);

  struct Converted {
    SimPortOutBase* out;
    SmallFunction<void(T&&, uint64_t), 48> send;
  };

  __attribute__((noinline)) uint32_t wire_size() const {
    SimPortOutBase* out = fwd_ ? static_cast<SimPortOutBase*>(fwd_) : conv_->out;
    SimPortInBase* end = out->endpoint();
    return end ? end->size() : 0;
  }

  // Drop the head: its slot, the credit and wake-ups it owes.
  void remove() {
    if (!kSimFunctional && reg_ready_) {
      auto& platform = SimPlatform::instance();
      if (freed_cycle_ != platform.cycles() || freed_gen_ != platform.reset_gen()) {
        freed_cycle_ = platform.cycles();
        freed_gen_ = platform.reset_gen();
        freed_count_ = 0;
      }
      ++freed_count_;
    }
    ring_.pop();
    --*inflight_;
    if (num_waiters_ != 0) {
      for (uint32_t i = 0; i < num_waiters_; ++i) {
        SimPlatform::instance().wake(waiters_[i]);
      }
      num_waiters_ = 0;
    }
    if (pop_cb_) {
      (*pop_cb_)();
    }
  }

  void bind_domain() {
    auto& dom = SimPlatform::instance().domain_at(owner_->domain());
    clock_ = &dom.now;
    head_ready_ = ring_.head_ready_ptr();
    inflight_ = &dom.inflight;
  }

  // The sender's capacity rule: room for one more packet now.
  __attribute__((always_inline)) bool has_room() const override {
    if constexpr (kSimFunctional) {
      return true;
    }
    auto& platform = SimPlatform::instance();
    uint32_t freed = (reg_ready_ && freed_cycle_ == platform.cycles() && freed_gen_ == platform.reset_gen())
                   ? freed_count_ : 0;
    return ring_.size() + freed < ring_.capacity();
  }

  // A producer found the port full: wake it when a slot frees. Under a
  // registered ready a slot popped this cycle frees next cycle, with no
  // further pop to announce it, so the waiter is then woken next cycle.
  void wait_for_room(SimTask* waiter) override {
    if (ring_.size() < ring_.capacity()) {
      SimPlatform::instance().wake_at(waiter, *clock_ + 1);
      return;
    }
    for (uint32_t i = 0; i < num_waiters_; ++i) {
      if (waiters_[i] == waiter) {
        return;
      }
    }
    __assert(num_waiters_ < kMaxWaiters, "too many producers waiting on one port");
    waiters_[num_waiters_++] = waiter;
  }

  // A packet sent toward this port. From another domain it is handed to
  // this port's domain as a deferred call, which re-issues it there next
  // cycle with one less latency, landing on the same cycle: a port's state
  // is touched only by its own domain. Cross-domain producers are credit-
  // gated at a boundary stage, so a deferred packet always finds room.
  void arrive(T&& pkt, uint64_t latency) {
    if constexpr (kSimFunctional) {
      latency = std::min<uint64_t>(latency, 1);
    }
    auto& platform = SimPlatform::instance();
    uint32_t domain = reader_->domain();
    if (platform.is_cross_exec(domain)) {
      __assert(latency >= 1, "a cross-domain send needs a registered latency");
      if constexpr (sizeof(T) <= 128) {
        platform.cross_call(domain, [this, p = std::move(pkt), latency]() {
          this->arrive(T(p), latency - 1);
        });
      } else {
        platform.cross_call(domain, [this, p = std::make_shared<T>(std::move(pkt)), latency]() {
          this->arrive(T(*p), latency - 1);
        });
      }
      return;
    }
    __assert(this->has_room(), "send on a full port");
    uint64_t ready = *clock_ + latency;
    ++*inflight_;
    ring_.insert(std::move(pkt), ready);
    // The reader wakes when the packet becomes visible. The functional kernel
    // also wakes it at the send, as its event delivery always did.
    if (ready > *clock_) {
      platform.wake_at(reader_, ready);
      if constexpr (kSimFunctional) {
        platform.wake(reader_);
      }
    } else {
      platform.wake(reader_);
    }
  }

  template <typename Call>
  static bool serve_with(SimPortIn* in, Call&& call) {
    if (in->empty()) {
      return false;
    }
    if (!call(in->ring_.front())) {
      return !detail::tl_blocked;   // retry next cycle, or when the output has room
    }
    in->remove();
    return !in->empty();   // one packet per cycle
  }

  static bool serve(SimPortInBase* base) {
    auto* in = static_cast<SimPortIn*>(base);
    return serve_with(in, [in](T& pkt) { return (in->owner_->*in->handler_)(pkt); });
  }

  static bool serve_indexed(SimPortInBase* base) {
    auto* in = static_cast<SimPortIn*>(base);
    return serve_with(in, [in](T& pkt) { return (in->owner_->*in->indexed_)(in->index_, pkt); });
  }

  static constexpr uint32_t kMaxWaiters = 4;

  // The ring header and the room check's state first: the hot line after
  // the base's.
  detail::DelayRing<T> ring_;
  int64_t* inflight_ = nullptr;
  bool reg_ready_ = false;
  uint32_t freed_count_ = 0;
  uint64_t freed_cycle_ = ~uint64_t(0);
  uint64_t freed_gen_ = 0;
  uint32_t num_waiters_ = 0;
  Handler handler_ = nullptr;
  IndexedHandler indexed_ = nullptr;
  uint32_t index_ = 0;
  SimTask* waiters_[kMaxWaiters] = {};
  SimPortOut<T>* fwd_ = nullptr;
  std::unique_ptr<Converted> conv_;
};

/// An output port, owned by the unit that writes it, bound once to a reader.
template <typename T>
class SimPortOut final : public SimPortOutBase {
public:
  explicit SimPortOut(SimObjectBase* owner) : SimPortOutBase(owner) {}

  /// Bind to a reader. `latency` is the one send(pkt) uses.
  void bind(SimPortIn<T>* sink, uint32_t latency = 1) {
    __assert(sink != nullptr, "bind target is null");
    __assert(sink_ == nullptr, "output already bound");
    sink_ = sink;
    latency_ = latency;
  }

  /// The reader has no room now. The running unit or handler is then woken
  /// when it has.
  __attribute__((always_inline)) bool full() {
    this->resolve();
    if (__builtin_expect(direct_ != nullptr ? direct_->has_room() : ep_->has_room(), 1)) {
      return false;
    }
    this->block();
    return true;
  }

  /// Send with the link's latency. Requires !full().
  void send(T pkt) { this->send(std::move(pkt), latency_); }

  /// Send with an explicit latency. Requires !full().
  void send(T pkt, uint32_t latency) {
    this->resolve();
    if (direct_ != nullptr) {
      direct_->arrive(std::move(pkt), latency);
    } else {
      conv_in_->conv_->send(std::move(pkt), latency);
    }
  }

  /// Send if the reader has room (as full()); returns whether it sent.
  /// The packet is copied only when it is sent.
  bool try_send(const T& pkt) { return this->try_send(pkt, latency_); }
  bool try_send(T&& pkt) { return this->try_send(std::move(pkt), latency_); }
  __attribute__((always_inline)) bool try_send(const T& pkt, uint32_t latency) {
    if (this->full()) {
      return false;
    }
    this->send(T(pkt), latency);
    return true;
  }
  __attribute__((always_inline)) bool try_send(T&& pkt, uint32_t latency) {
    if (this->full()) {
      return false;
    }
    this->send(std::move(pkt), latency);
    return true;
  }

  /// Packets sent and not yet taken by the reader, and the reader's room.
  uint32_t size() const { this->resolve(); return ep_->size(); }
  uint32_t capacity() const { this->resolve(); return ep_->capacity(); }

  SimPortInBase* endpoint() override {
    if (sink_ == nullptr) {
      return nullptr;
    }
    this->resolve();
    return ep_;
  }

private:
  // Follow the wires to the reader once, on first use: bindings are fixed
  // by then.
  void resolve() const {
    if (__builtin_expect(ep_ == nullptr, 0)) {
      this->resolve_first();
    }
  }

  // The reader has no room: wake the running unit or handler when it has.
  __attribute__((noinline, cold)) void block() {
    ep_->wait_for_room(SimPlatform::instance().running_task());
    detail::tl_blocked = true;
  }

  __attribute__((noinline, cold)) void resolve_first() const {
    __assert(sink_ != nullptr, "send on an unbound output");
    SimPortIn<T>* in = sink_;
    while (in->fwd_ != nullptr) {
      __assert(in->fwd_->sink_ != nullptr, "a wire to an unbound output");
      in = in->fwd_->sink_;
    }
    if (in->conv_) {
      conv_in_ = in;
      ep_ = in->conv_->out->endpoint();
    } else {
      direct_ = in;
      ep_ = in;
    }
  }

  SimPortIn<T>* sink_ = nullptr;
  mutable SimPortIn<T>* direct_ = nullptr;    // the reader, when no conversion is on the way
  mutable SimPortIn<T>* conv_in_ = nullptr;   // otherwise, the converting wire
  mutable SimPortInBase* ep_ = nullptr;
  uint32_t latency_ = 1;
};

namespace detail {

[[noreturn]] __attribute__((noinline, cold)) inline void port_index_fault(size_t i, size_t n) {
  throw std::out_of_range("port index " + std::to_string(i) + " >= " + std::to_string(n));
}

// at(i) of a port array: checked like std::vector::at, with the fault path
// out of line so the check inlines.
inline void check_port_index(size_t i, size_t n) {
  if (__builtin_expect(i >= n, 0)) {
    port_index_fault(i, n);
  }
}

// Iterates a vector of owned ports by reference.
template <typename P, typename It>
class PortIter {
public:
  explicit PortIter(It it) : it_(it) {}
  P& operator*() const { return **it_; }
  PortIter& operator++() { ++it_; return *this; }
  bool operator!=(const PortIter& other) const { return it_ != other.it_; }

private:
  It it_;
};

}  // namespace detail

/// N input ports of one unit.
template <typename T>
class SimPortInArray {
public:
  SimPortInArray(SimObjectBase* owner, uint32_t count, uint32_t capacity = 2) {
    for (uint32_t i = 0; i < count; ++i) {
      ports_.emplace_back(new SimPortIn<T>(owner, capacity));
    }
  }

  /// N handler ports: `handler(i, pkt)` serves port i.
  template <typename U>
  SimPortInArray(U* owner, uint32_t count, uint32_t capacity, bool (U::*handler)(uint32_t, T&)) {
    for (uint32_t i = 0; i < count; ++i) {
      ports_.emplace_back(new SimPortIn<T>(owner, capacity, handler, i));
    }
  }

  SimPortIn<T>& operator[](uint32_t i) { return *ports_[i]; }
  const SimPortIn<T>& operator[](uint32_t i) const { return *ports_[i]; }
  SimPortIn<T>& at(uint32_t i) { detail::check_port_index(i, ports_.size()); return *ports_[i]; }
  const SimPortIn<T>& at(uint32_t i) const { detail::check_port_index(i, ports_.size()); return *ports_[i]; }
  auto begin() { return detail::PortIter<SimPortIn<T>, typename Ports::iterator>(ports_.begin()); }
  auto end() { return detail::PortIter<SimPortIn<T>, typename Ports::iterator>(ports_.end()); }
  auto begin() const { return detail::PortIter<const SimPortIn<T>, typename Ports::const_iterator>(ports_.begin()); }
  auto end() const { return detail::PortIter<const SimPortIn<T>, typename Ports::const_iterator>(ports_.end()); }
  uint32_t size() const { return uint32_t(ports_.size()); }
  bool empty() const { return ports_.empty(); }

  /// Bit k set when port first + k has a visible packet (count <= 64).
  uint64_t ready_mask(uint32_t first, uint32_t count) const {
    uint64_t mask = 0;
    uint32_t end = std::min<uint32_t>(first + count, uint32_t(ports_.size()));
    for (uint32_t i = first; i < end; ++i) {
      mask |= uint64_t(!ports_[i]->empty()) << (i - first);
    }
    return mask;
  }

  uint64_t ready_mask() const { return this->ready_mask(0, this->size()); }

private:
  using Ports = std::vector<std::unique_ptr<SimPortIn<T>>>;
  Ports ports_;
};

/// N output ports of one unit.
template <typename T>
class SimPortOutArray {
public:
  SimPortOutArray(SimObjectBase* owner, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
      ports_.emplace_back(new SimPortOut<T>(owner));
    }
  }

  SimPortOut<T>& operator[](uint32_t i) { return *ports_[i]; }
  const SimPortOut<T>& operator[](uint32_t i) const { return *ports_[i]; }
  SimPortOut<T>& at(uint32_t i) { detail::check_port_index(i, ports_.size()); return *ports_[i]; }
  const SimPortOut<T>& at(uint32_t i) const { detail::check_port_index(i, ports_.size()); return *ports_[i]; }
  auto begin() { return detail::PortIter<SimPortOut<T>, typename Ports::iterator>(ports_.begin()); }
  auto end() { return detail::PortIter<SimPortOut<T>, typename Ports::iterator>(ports_.end()); }
  auto begin() const { return detail::PortIter<const SimPortOut<T>, typename Ports::const_iterator>(ports_.begin()); }
  auto end() const { return detail::PortIter<const SimPortOut<T>, typename Ports::const_iterator>(ports_.end()); }
  uint32_t size() const { return uint32_t(ports_.size()); }
  bool empty() const { return ports_.empty(); }

private:
  using Ports = std::vector<std::unique_ptr<SimPortOut<T>>>;
  Ports ports_;
};

inline void SimObjectBase::tick_wake() {
  auto& platform = SimPlatform::instance();
  if (ticks_) {
    platform.wake(this);
  }
  for (SimPortInBase* in : handlers_) {
    platform.wake(in->entry_.get());
  }
}

inline bool SimObjectBase::inputs_ready() const {
  for (const SimPortInBase* in : plain_) {
    if (in->ready_now()) {
      return true;
    }
  }
  return false;
}

///////////////////////////////////////////////////////////////////////////////
// Platform implementation
///////////////////////////////////////////////////////////////////////////////

template <typename U, typename... Args>
std::shared_ptr<U> SimPlatform::create_object(Args&&... args) {
  auto obj = std::make_shared<U>(SimContext{}, std::forward<Args>(args)...);
  objects_.push_back(obj);
  // A unit with an on_tick() joins the scan as it is created: after its ports
  // and the units its constructor made.
  if constexpr (U::kTicks) {
    this->add_scan(obj.get());
  }
  if constexpr (U::kResets) {
    resets_.push_back(obj.get());
  }
  return obj;
}

inline void SimPlatform::sync_init_graph() {
  uint32_t n = num_domains_;
  sync_edge_.assign(size_t(n) * n, 0);
  auto link = [&](uint32_t a, uint32_t b) {
    if (a != b) {
      sync_edge_[size_t(a) * n + b] = 1;
      sync_edge_[size_t(b) * n + a] = 1;
    }
  };
  for (auto* out : outs_) {
    if (auto* ep = out->endpoint()) {
      link(out->owner_->domain(), ep->owner_->domain());
    }
  }
  for (uint32_t d = 1; d < n; ++d) {
    link(0, d);
  }
  sync_deps_.assign(n, {});
  for (uint32_t a = 0; a < n; ++a) {
    for (uint32_t b = 0; b < n; ++b) {
      if (sync_edge_[size_t(a) * n + b]) {
        sync_deps_[a].push_back(b);
      }
    }
  }
  sync_progress_.reset(new SyncProgress[n]);
}

// At the first reset: every output that crosses an execution-domain boundary
// must do so through a registered boundary stage. An unregistered crossing
// is safe serially (its sends go through the mailboxes) but its capacity
// check reads another domain's port, so it is reported once and execution
// stays on one thread.
inline void SimPlatform::validate_topology() {
  if (topo_validated_) {
    return;
  }
  topo_validated_ = true;
  if (num_domains_ < 2) {
    return;
  }
  bool clamped = false;
  for (auto* out : outs_) {
    auto* ep = out->endpoint();
    if (ep == nullptr || out->boundary_ || ep->owner_->domain() == out->owner_->domain()) {
      continue;
    }
    std::cout << "SIMX-MT: unregistered cross-domain edge: "
              << out->owner_->name() << " -> " << ep->owner_->name() << std::endl;
    clamped = true;
  }
  if (clamped) {
    std::cout << "SIMX-MT: parallel execution disabled until every "
                 "cross-domain edge carries a registered stage" << std::endl;
    mt_max_threads_ = 1;
  }
}

// A kernel launch resets every unit but those built under a LaunchKeepScope:
// the hardware does not reset between a transfer and the launch that follows
// it, so the transfer's line fills may still be in flight in the kept memory
// side when the kernel starts.
inline void SimPlatform::reset_launch() {
  launch_resetting_ = true;
  for (auto* unit : resets_) {
    if (!unit->launch_kept_) {
      unit->do_reset();
    }
  }
  launch_resetting_ = false;
  this->wake_all();
}

// Callers reset only from quiescence (see idle()).
inline void SimPlatform::reset() {
  this->validate_topology();
  for (auto* unit : resets_) {
    unit->do_reset();
  }
  this->wake_all();
  // The lanes themselves are kept: workers index them once started.
  this->clear_cross_lanes();
  cycles_ = 0;
  for (auto& dom : domains_) {
    dom->now = 0;
    for (auto& slot : dom->wake_slots) {
      std::fill(slot.begin(), slot.end(), 0);
    }
    dom->wake_armed = 0;
    dom->wake_far.clear();
  }
  ++reset_gen_;
}

inline void SimPlatform::tick() {
  if (!mt_init_done_) {
    this->init_mt(req_workers_);
  }
  if (workers_started_) {
    // Workers park on the entry barrier between tick() calls, so everything
    // outside this function is single-threaded.
    this->barrier_wait(0);
    this->run_stripe(0);
    this->barrier_wait(0, [this]() { this->cycle_end(); });
    return;
  }
  // Serial: domains interact only through registered (latency >= 1) edges,
  // so their order within a cycle is not observable.
  for (auto& dom : domains_) {
    this->run_domain(*dom, cycles_);
  }
  cycles_ += quantum_;
  tl_exec_domain_ = HOST_DOMAIN;
  tl_exec_task_ = nullptr;
}

template <typename Stop>
void SimPlatform::run_until(Stop&& stop) {
  if (!mt_init_done_) {
    this->init_mt(req_workers_);
  }
  if (!workers_started_) {
    do {
      this->tick();
    } while (!stop());
    return;
  }
  __assert(!run_active_, "run_until() is not reentrant");
  using StopFn = std::remove_reference_t<Stop>;
  run_stop_ = [](void* arg) -> bool { return (*static_cast<StopFn*>(arg))(); };
  run_stop_arg_ = const_cast<void*>(static_cast<const void*>(&stop));
  // Set in the release step: a worker reads the flag after each cycle-end
  // barrier, and one slow to leave the previous run's last barrier must not
  // see this run's value.
  this->barrier_wait(0, [this]() { run_active_ = true; });
  do {
    this->run_stripe(0);
    this->barrier_wait(0, [this]() { this->cycle_end(); });
  } while (run_active_);
  run_stop_ = nullptr;
  run_stop_arg_ = nullptr;
}

template <typename Stop, typename Obj, typename Busy>
void SimPlatform::run_until(Stop&& stop, const std::vector<Obj*>& voters, Busy&& busy) {
  if (!mt_init_done_) {
    this->init_mt(req_workers_);
  }
  if (!workers_started_) {
    do {
      this->tick();
    } while (!stop(false));
    return;
  }
  __assert(!run_active_ && !sync_active_, "run_until() is not reentrant");
  __assert(stripe_domains_[0].front()->id == 0, "thread 0 owns the uncore domain");
  if (sync_progress_ == nullptr) {
    this->sync_init_graph();
  }
  std::vector<std::vector<Obj*>> domain_voters(num_domains_);
  for (auto* voter : voters) {
    domain_voters.at(voter->domain()).push_back(voter);
  }
  auto vote = [&](uint32_t domain) {
    for (auto* voter : domain_voters[domain]) {
      if (busy(voter)) {
        return true;
      }
    }
    return false;
  };
  using StopFn = std::remove_reference_t<Stop>;
  using VoteFn = decltype(vote);
  sync_stop_fn_ = [](void* arg, bool any_busy) -> bool { return (*static_cast<StopFn*>(arg))(any_busy); };
  sync_vote_fn_ = [](void* arg, uint32_t d) -> bool { return (*static_cast<VoteFn*>(arg))(d); };
  sync_stop_arg_ = const_cast<void*>(static_cast<const void*>(&stop));
  sync_vote_arg_ = const_cast<void*>(static_cast<const void*>(&vote));
  sync_start_ = cycles_;
  for (uint32_t d = 0; d < num_domains_; ++d) {
    sync_progress_[d].word.store(sync_start_ << 1, std::memory_order_relaxed);
  }
  sync_decided_.store(sync_start_, std::memory_order_relaxed);
  sync_stop_.store(false, std::memory_order_relaxed);
  sync_active_ = true;
  this->barrier_wait(0);
  this->sync_loop(0);
  this->barrier_wait(0);
  sync_active_ = false;
  sync_stop_fn_ = nullptr;
  sync_vote_fn_ = nullptr;
  sync_stop_arg_ = nullptr;
  sync_vote_arg_ = nullptr;
}

inline void SimPlatform::tick_domain(Domain& dom, bool drain) {
  dom.now = tl_cycles_;
  // timed wake-ups due this cycle join the ready set
  uint32_t slot = uint32_t(dom.now & (Domain::kWakeSlots - 1));
  if ((dom.wake_armed >> slot) & 1) {
    auto& bits = dom.wake_slots[slot];
    for (uint32_t w = 0; w < dom.active_words; ++w) {
      dom.active()[w] |= bits[w];
      bits[w] = 0;
    }
    dom.wake_armed &= ~(uint64_t(1) << slot);
  }
  while (!dom.wake_far.empty() && dom.wake_far.front().first <= dom.now) {
    uint32_t idx = dom.wake_far.front().second;
    dom.active()[idx >> 6] |= uint64_t(1) << (idx & 63);
    std::pop_heap(dom.wake_far.begin(), dom.wake_far.end(), std::greater<>());
    dom.wake_far.pop_back();
  }
  tl_exec_domain_ = dom.id;
  tl_exec_task_ = nullptr;
  if (drain) {
    this->drain_cross_calls(dom.id);
  }
  // Visit the ready set in creation order. The word is re-read after each
  // task, so a task woken by one ahead of it runs this cycle and one woken by
  // a task behind it runs next cycle, as in a scan of every task that tests
  // a flag.
  uint64_t* words = dom.active_words ? dom.active() : nullptr;
  SimTask* const* tasks = dom.scan.data();
  for (uint32_t w = 0, nw = dom.active_words; w < nw; ++w) {
    uint64_t bits = words[w];
    while (bits != 0) {
      uint32_t b = uint32_t(__builtin_ctzll(bits));
      auto* task = tasks[size_t(w) * 64 + b];
      tl_exec_task_ = task;
      task->run();
      bits = words[w] & (~uint64_t(1) << b);
    }
  }
}

inline void SimPlatform::cleanup() {
  this->stop_workers();
  resets_.clear();
  outs_.clear();
  objects_.clear();
  domains_.clear();
  mail_.clear();
  sync_progress_.reset();
  sync_deps_.clear();
  sync_edge_.clear();
  num_domains_ = 1;
  cycles_ = 0;
  // A later initialize() elaborates and parallelizes afresh.
  mt_init_done_ = false;
  mt_max_threads_ = std::numeric_limits<uint32_t>::max();
  topo_validated_ = false;
}

#ifdef SIMX_FUNCTIONAL
}  // inline namespace sim_functional
#endif

}  // namespace vortex
