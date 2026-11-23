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

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "linked_list.h"
#include "mempool.h"
#include "util.h"
#include "smallfunc.h"

// The platform's per-thread execution context is read on every send and
// cycles() query. initial-exec turns each access into one %fs-relative load
// instead of a __tls_get_addr call. It also puts libsimx's whole TLS segment
// in glibc's static block, whose spare room for dlopen'ed libraries is a few
// hundred bytes shared by all of them: keep thread_local state in libsimx to
// a few pointers.
#ifndef SIMX_TLS_IE
#define SIMX_TLS_IE __attribute__((tls_model("initial-exec")))
#endif
#include "ringqueue.h"

namespace vortex {

// Forward declarations
class SimPlatform;
class SimObjectBase;
template <typename T> class SimChannel;
template <typename T> class SimEventLink;
template <typename T> class RegSlice;

class SimContext {
private:
  SimContext() = default;
  friend class SimPlatform;
  template <typename T> friend class SimChannel;
};

// Base class for all simulation objects (Nodes)
class SimObjectBase {
public:
  using Ptr = std::shared_ptr<SimObjectBase>;
  virtual ~SimObjectBase() = default;
  // An object's ready-set binding is its own: objects are never copied.
  SimObjectBase(const SimObjectBase&) = delete;
  SimObjectBase& operator=(const SimObjectBase&) = delete;

  const std::string& name() const { return name_; }

  // Execution domain this object was created under (see SimPlatform::DomainScope).
  // Domain 0 is the uncore/default domain; topology containers open one scope
  // per partition. Leaf modules never set this — they inherit the build scope.
  uint32_t domain() const { return domain_; }

protected:
  // Defined after SimPlatform: tags the object with the build-domain scope in
  // effect at construction (covers by-value member SimObjects that never pass
  // through create_object).
  SimObjectBase(const SimContext&, const std::string& name);

  // Tick gating: an object whose on_tick() is provably a no-op in its current
  // state (all watched input channels empty, nothing in flight toward them)
  // may drop out of the per-cycle tick scan. Reserving a packet toward one of
  // its endpoint channels re-arms it before delivery, so a skipped tick never
  // hides work; so does a packet toward a channel it watches (see
  // SimChannelBase::wake_on_push) and a pop from one it waits on
  // (wake_on_pop). Objects opt in by calling tick_sleep() from on_tick();
  // tick_wake() re-arms from non-channel entry points (e.g. a flush request).
  void tick_sleep() { *active_word_ &= ~active_bit_; }
  void tick_wake()  { *active_word_ |= active_bit_; }

private:
  std::string name_;
  // This object's bit in its domain's ready set (see SimPlatform::Domain);
  // an object without a tick points at its own scratch word.
  uint64_t* active_word_ = &inert_word_;
  uint64_t active_bit_ = 1;
  uint64_t inert_word_ = 0;
  uint32_t domain_ = 0;
  bool launch_kept_ = false; // built under a LaunchKeepScope
  virtual void do_reset() = 0;
  virtual void do_tick()  = 0;
  friend class SimPlatform;
  friend class SimChannelBase;
};

// Base class for channels (Topological introspection)
class SimChannelBase {
public:
  virtual ~SimChannelBase();

  // Introspection API
  SimObjectBase* module() const { return module_; }
  SimChannelBase* sink() const { return sink_; }
  SimChannelBase* source() const { return source_; }

  // Declares this channel as the outgoing side of a registered boundary
  // stage. Topology validation requires a marked channel on every chain that
  // crosses an execution-domain boundary; only boundary stages call this.
  void mark_boundary() { boundary_stage_ = true; }

  // Invoked on every endpoint pop; a boundary stage installs this on its
  // downstream endpoint to return a flow-control credit.
  void pop_callback(SmallFunction<void(), 48>&& callback) {
    pop_cb_ = std::move(callback);
  }

  // Wake `reader` too when a packet is sent toward this endpoint: for an
  // object that reads a channel another object owns (a sub-object's output)
  // and sleeps while it is empty. Bindings are made after elaboration binds
  // the chain, on the endpoint itself, in the same execution domain.
  void wake_on_push(SimObjectBase* reader) {
    this->bind_waker(&push_waker_, reader);
  }

  // Wake `writer` when a packet is popped from this endpoint: for a producer
  // that sleeps while the channel is full. The pop frees the slot the
  // producer waits on (the next cycle, under a registered ready).
  void wake_on_pop(SimObjectBase* writer) {
    this->bind_waker(&pop_waker_, writer);
  }

  virtual bool empty() const = 0;
  virtual bool full() const = 0;
  virtual uint32_t size() const = 0;
  virtual uint32_t capacity() const = 0;

protected:
  // Defined after SimPlatform: registers with the platform channel registry
  // (used for topology validation).
  explicit SimChannelBase(SimObjectBase* module);

  virtual void reserve() = 0;

  // Re-arm a tick-gated module (see SimObjectBase::tick_sleep).
  static void wake_module(SimObjectBase* module) {
    module->tick_wake();
  }

  void bind_waker(SimObjectBase** slot, SimObjectBase* obj) {
    __assert(obj != nullptr, "waker is null");
    __assert(sink_ == nullptr, "a waker binds to a channel endpoint");
    __assert(obj->domain() == module_->domain(), "a waker shares the endpoint's domain");
    __assert(*slot == nullptr || *slot == obj, "channel waker already bound");
    *slot = obj;
  }

  SimObjectBase*  module_;
  SimObjectBase*  push_waker_ = nullptr;
  SimObjectBase*  pop_waker_ = nullptr;
  SimChannelBase* sink_;
  SimChannelBase* source_;
  SimChannelBase* endpoint_cache_ = nullptr;
  // Packets sent toward this endpoint and not yet consumed are counted in the
  // endpoint's domain (bound at construction): only that domain's context
  // touches the counter, so it needs no atomic and shares no cache line.
  int64_t* inflight_ = nullptr;
  int64_t& inflight() { return *inflight_; }
  int64_t& resolve_inflight();
  // The endpoint domain's cycle counter (bound at construction): a delay-line
  // endpoint compares its head's ready cycle against it.
  const uint64_t* clock_ = nullptr;
  const uint64_t& clock() const { return *clock_; }
  const uint64_t& resolve_clock();
  // Bind the domain counters (at construction, and when a copy-assignment
  // moves the channel to another module).
  void resolve_domain() {
    inflight_ = &this->resolve_inflight();
    clock_ = &this->resolve_clock();
  }
  // A send writes straight into the endpoint's delay line unless a delivery
  // callback sits on the chain, which must run at arrival (resolved on first
  // send; 0 unknown, 1 direct, 2 through events).
  bool has_tx_cb_ = false;
  uint8_t delivery_mode_ = 0;
  bool chain_has_tx_cb() const {
    for (auto* e = this; e != nullptr; e = e->sink_) {
      if (e->has_tx_cb_) {
        return true;
      }
    }
    return false;
  }
  bool boundary_stage_ = false;
  SmallFunction<void(), 48> pop_cb_;

  // Terminal endpoint of the bind chain (cached: bindings are immutable
  // after construction).
  // Bindings are immutable after elaboration, so the endpoint is resolved
  // once and occupancy queries go straight to it.
  SimChannelBase* endpoint() const {
    return const_cast<SimChannelBase*>(this)->resolve_endpoint();
  }

  SimChannelBase* resolve_endpoint() {
    if (endpoint_cache_ == nullptr) {
      auto* e = this;
      while (e->sink_) {
        e = e->sink_;
      }
      endpoint_cache_ = e;
    }
    return endpoint_cache_;
  }

  friend class SimPlatform;
  template <typename T> friend class SimChannel;
};

// Events
///////////////////////////////////////////////////////////////////////////////
class SimEventBase {
public:
  virtual ~SimEventBase() = default;
  virtual void fire() = 0;
  uint64_t cycles() const { return cycles_; }

protected:
  explicit SimEventBase(uint64_t cycles) : cycles_(cycles) {}
  uint64_t cycles_;
  LinkedListNode<SimEventBase> list_;
  friend class SimPlatform;
};

// Optimized Event for Channel Transfers
template <typename Pkt>
class SimChannelEvent final : public SimEventBase {
public:
  template <typename P>
  SimChannelEvent(SimChannel<Pkt>* channel, P&& pkt, uint64_t cycles)
      : SimEventBase(cycles), channel_(channel), pkt_(std::forward<P>(pkt)) {}

  void fire() override;

  static void* operator new(std::size_t) { return allocator_.allocate(1); }
  static void operator delete(void* ptr, std::size_t) noexcept {
    allocator_.deallocate(static_cast<SimChannelEvent<Pkt>*>(ptr), 1);
  }

private:
  SimChannel<Pkt>* channel_;
  Pkt pkt_;
  static inline PoolAllocator<SimChannelEvent<Pkt>, 128> allocator_;
};

// Generic Event for arbitrary callbacks
template <typename Pkt>
class SimCallEvent final : public SimEventBase {
public:
  using Func = SmallFunction<void(const Pkt&), 48>;

  template <typename F>
  SimCallEvent(F&& func, Pkt pkt, uint64_t cycles)
    : SimEventBase(cycles), func_(std::forward<F>(func)), pkt_(std::move(pkt)) {}

  void fire() override { func_(pkt_); }

  static void* operator new(std::size_t) { return allocator_.allocate(1); }
  static void operator delete(void* ptr, std::size_t) noexcept {
    allocator_.deallocate(static_cast<SimCallEvent<Pkt>*>(ptr), 1);
  }

private:
  Func func_;
  Pkt pkt_;
  static inline PoolAllocator<SimCallEvent<Pkt>, 64> allocator_;
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
  void finalize() { cleanup(); }

  // In domain context this is the executing phase's cycle (delivery runs one
  // cycle ahead of tick): thread-local, so workers advance through a cycle's
  // phases without a global synchronization point. Host context sees the
  // committed global counter.
  uint64_t cycles() const {
    return (tl_exec_domain_ == HOST_DOMAIN) ? cycles_ : tl_cycles_;
  }

  // Counts resets: cycle numbers repeat across runs, so per-cycle state is
  // tagged with the run it belongs to.
  uint64_t reset_gen() const { return reset_gen_; }

  // Sends issued outside any object tick (driver/host calls).
  static constexpr uint32_t HOST_DOMAIN = 0xffffffffu;

  // True when the platform holds no undelivered work: no packet sent but not
  // yet consumed, and no deferred cross-domain delivery in flight. This is
  // the only valid quiescence test — host loops must not reassemble it from
  // framework internals.
  bool idle() const {
    int64_t inflight = 0;
    for (auto& dom : domains_) {
      if (dom) {
        inflight += dom->inflight;
      }
    }
    return inflight == 0 && this->cross_pending() == 0;
  }

  // Domain topology API — used only by topology containers (Socket/Cluster);
  // leaf modules inherit the scope in effect when they are created.
  uint32_t alloc_domain() { return num_domains_++; }
  uint32_t num_domains() const { return num_domains_; }
  uint32_t build_domain() const { return build_domain_; }
  bool build_launch_kept() const { return build_launch_kept_; }

  // True while reset_launch() runs: an object resets its launch state but keeps
  // what only a device reset clears.
  bool launch_resetting() const { return launch_resetting_; }

  // Cap the parallel thread count (must be called before the first tick).
  // A limit of 1 forces serial execution — required while the configuration
  // contains a cross-domain channel edge that is not registered.
  void mt_limit(uint32_t max_threads) {
    mt_max_threads_ = std::max(1u, max_threads);
  }

  // Request the worker-thread count for the lockstep executor (must be
  // called before the first tick); 0 or 1 selects serial execution. Cycle
  // results are bit-identical for every value.
  void set_num_workers(uint32_t num_workers) {
    req_workers_ = std::max(1u, num_workers);
  }

  // The timed kernel keeps per-cycle lockstep: its cross-domain latencies are
  // physical, so only the functional kernel accepts a larger quantum.
  void set_quantum(uint32_t cycles) {
    __assert(cycles <= 1, "a synchronization quantum requires the functional kernel");
  }

  // Objects built under a LaunchKeepScope (the last-level cache and the
  // memory below it, when the command processor writes into that cache)
  // carry their state and in-flight work across reset_launch().
  class LaunchKeepScope {
  public:
    explicit LaunchKeepScope(bool keep) {
      auto& p = SimPlatform::instance();
      saved_ = p.build_launch_kept_;
      p.build_launch_kept_ = saved_ || keep;
    }
    ~LaunchKeepScope() {
      SimPlatform::instance().build_launch_kept_ = saved_;
    }
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
    // Open the partition an existing module lives in — used when a boundary
    // stage must be owned by a peer's domain rather than the current scope.
    explicit DomainScope(const SimObjectBase* module)
      : DomainScope(module->domain()) {}
    ~DomainScope() {
      SimPlatform::instance().build_domain_ = saved_;
    }
    DomainScope(const DomainScope&) = delete;
    DomainScope& operator=(const DomainScope&) = delete;
  private:
    uint32_t saved_;
  };

  // Boundary audit (VX_SIMX_AUDIT_BOUNDARY=1): histogram every cross-domain
  // send as (exec domain -> endpoint domain, delay, endpoint) so a partition
  // cut can be checked for illegal zero-delay crossings.
  bool audit_enabled() const { return audit_enabled_; }
  void audit_send(SimChannelBase* channel, uint64_t delay);
  void dump_audit(std::ostream& os);

  // True when the currently executing context must not touch this module's
  // state directly (it belongs to a different domain and we're inside a
  // domain tick, not host context).
  bool is_cross_exec(const SimObjectBase* module) const {
    return tl_exec_domain_ != HOST_DOMAIN
        && module->domain() != tl_exec_domain_;
  }

  // True when a parallel worker is touching another domain's channel state —
  // an unsynchronized access that parallel execution forbids.
  bool mt_cross_access(const SimObjectBase* module) const {
    return mt_threads_ > 1 && this->is_cross_exec(module);
  }

  // Factory
  template <typename Impl, typename... Args>
  std::shared_ptr<Impl> create_object(Args&&... args);

  // Scheduling API. The callback overload requires the domain whose state
  // the callback mutates (it fires on that domain's wheel).
  template <typename Pkt, typename Func>
  void schedule(Func&& func, const Pkt& pkt, uint64_t delay, uint32_t domain);

  void reset();
  void reset_launch();
  void tick();

  // Tick until stop() returns true; stop() is evaluated in host context after
  // every cycle, as in `do { tick(); } while (!stop());`, which it equals
  // cycle for cycle. With workers, the check runs in the release step of the
  // cycle-end barrier, so a cycle costs one barrier instead of tick()'s two.
  template <typename Stop>
  void run_until(Stop&& stop);

  // Tick until stop(busy) returns true, for a loop that cannot end while any
  // object of `voters` is busy(voter). After each cycle the thread ticking a
  // voter evaluates it, and stop runs with busy set when one was: it must then
  // return false and touch only uncore state, since other objects may be
  // ticking. stop(false) sees exactly the state run_until(stop) would. With
  // workers, neighbor synchronization replaces the barrier (sync_loop).
  template <typename Stop, typename Obj, typename Busy>
  void run_until(Stop&& stop, const std::vector<Obj*>& voters, Busy&& busy);

private:
  // Timing Wheel Configuration
  static constexpr uint64_t WHEEL_SIZE = 4096;
  static constexpr uint64_t WHEEL_MASK = WHEEL_SIZE - 1;

  // A lane event: a small trivially copyable packet rides inline, anything
  // else as a pooled event object. fn fires the event (run) or drops it.
  static constexpr size_t kLanePkt = 16;
  struct LaneEvent {
    void (*fn)(LaneEvent&, bool run);
    void* target;
    alignas(8) unsigned char pkt[kLanePkt];
  };

  template <typename Pkt>
  static constexpr bool lane_inline() {
    return std::is_trivially_copyable_v<Pkt> && sizeof(Pkt) <= kLanePkt && alignof(Pkt) <= 8;
  }

  template <typename Pkt>
  static void lane_fire_inline(LaneEvent& e, bool run) {
    if (run) {
      static_cast<SimChannel<Pkt>*>(e.target)->receive_packet(*std::launder(reinterpret_cast<const Pkt*>(e.pkt)));
    }
  }

  static void lane_fire_event(LaneEvent& e, bool run) {
    auto* evt = static_cast<SimEventBase*>(e.target);
    if (run) {
      evt->fire();
    }
    delete evt;
  }

  template <typename Pkt, typename P>
  static void lane_push(std::vector<LaneEvent>& lane, SimChannel<Pkt>* channel, P&& pkt) {
    LaneEvent e;
    e.target = channel;
    if constexpr (lane_inline<Pkt>()) {
      e.fn = &lane_fire_inline<Pkt>;
      new (e.pkt) Pkt(std::forward<P>(pkt));
    } else {
      e.fn = &lane_fire_event;
      e.target = static_cast<SimEventBase*>(new SimChannelEvent<Pkt>(channel, std::forward<P>(pkt), 0));
    }
    lane.push_back(e);
  }

  static void lane_drop(std::vector<LaneEvent>& lane) {
    for (auto& e : lane) {
      e.fn(e, false);
    }
    lane.clear();
  }

  // Per-domain kernel state: scan list, timing wheel, and immediate-event
  // list are private to one execution domain, so a domain can be ticked by
  // its own worker without touching another domain's structures. Events are
  // routed to the ENDPOINT's domain at schedule time.
  struct Domain {
    std::vector<SimObjectBase*> scan;   // creation order within the domain
    // Ready set: bit i is set while scan[i] is awake. Objects sleep and wake
    // by clearing and setting their own bit, so a cycle visits only the
    // awake ones, still in creation order. The words come in cache-line
    // blocks: each domain's are written by its own worker only.
    struct alignas(64) ActiveBlock {
      uint64_t words[8] = {};
    };
    std::vector<ActiveBlock> active_blocks;
    uint32_t active_words = 0;
    uint64_t* active() { return active_blocks.data()->words; }
    std::vector<LinkedList<SimEventBase, &SimEventBase::list_>> wheel;
    // Events due next cycle (delay 1), by the due cycle's parity, and this
    // cycle (delay 0), each in schedule order. A delay-1 event is scheduled
    // after every longer-delay event due the same cycle (a domain's time only
    // moves forward), so firing the wheel bucket and then the lane keeps the
    // order a single wheel gives.
    std::vector<LaneEvent> next[2];
    std::vector<LaneEvent> imm;
    // Packets in flight toward this domain's endpoints (see SimChannelBase).
    alignas(64) int64_t inflight = 0;
    // The cycle this domain is ticking, or delivering into; the next cycle to
    // run between cycles. Read by its delay-line endpoints.
    uint64_t now = 0;
    uint32_t id = 0;
    Domain() : wheel(WHEEL_SIZE) {}
  };

  Domain& domain_at(uint32_t d) {
    if (__builtin_expect(d < domains_.size(), 1)) {
      if (auto* dom = domains_[d].get()) {
        return *dom;
      }
    }
    return this->domain_grow(d);
  }

  __attribute__((noinline)) Domain& domain_grow(uint32_t d) {
    if (domains_.size() <= d) {
      domains_.resize(d + 1);
      for (uint32_t i = 0; i < domains_.size(); ++i) {
        if (!domains_[i]) {
          domains_[i] = std::make_unique<Domain>();
          domains_[i]->id = i;
        }
      }
    }
    return *domains_.at(d);
  }

  // Elaboration only: register a ticking object in its domain's scan list.
  void add_scan(SimObjectBase* object) {
    auto& dom = this->domain_at(object->domain_);
    uint32_t idx = uint32_t(dom.scan.size());
    dom.scan.push_back(object);
    if ((idx & 63) == 0) {
      if ((dom.active_words & 7) == 0) {
        auto* before = dom.active_blocks.data();
        dom.active_blocks.emplace_back();
        if (dom.active_blocks.data() != before) {
          // the words moved: re-point every object at its word
          for (uint32_t i = 0; i < idx; ++i) {
            dom.scan[i]->active_word_ = &dom.active()[i >> 6];
          }
        }
      }
      ++dom.active_words;
    }
    object->active_word_ = &dom.active()[idx >> 6];
    object->active_bit_ = uint64_t(1) << (idx & 63);
    *object->active_word_ |= object->active_bit_;
  }

  // Wake every ticking object (a reset re-arms them all).
  void wake_all() {
    for (auto& dom : domains_) {
      uint32_t n = uint32_t(dom->scan.size());
      for (uint32_t w = 0; w < dom->active_words; ++w) {
        uint32_t bits = std::min(64u, n - w * 64);
        dom->active()[w] = (bits == 64) ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1);
      }
    }
  }

  SimPlatform() : cycles_(0) {
    audit_requested_ = []() {
      const char* env = std::getenv("VX_SIMX_AUDIT_BOUNDARY");
      return env != nullptr && env[0] == '1';
    }();
    audit_enabled_ = audit_requested_;
  }
  ~SimPlatform() { cleanup(); }

  void cleanup();
  void tick_domain(Domain& dom);
  void fire_immediate_events(Domain& dom);

  template <typename Pkt>
  void schedule(SimChannel<Pkt>* channel, const Pkt& pkt, uint64_t delay);

  template <typename Pkt>
  void schedule(SimChannel<Pkt>* channel, Pkt&& pkt, uint64_t delay);

  std::vector<std::shared_ptr<SimObjectBase>> objects_;
  // Only objects that override on_reset() (auto-detected at create_object).
  std::vector<SimObjectBase*> active_reset_;
  // Per-domain scan/wheel/imm state; objects that override on_tick() are
  // registered in their domain's scan list.
  std::vector<std::unique_ptr<Domain>> domains_;

  uint64_t cycles_;
  uint64_t reset_gen_ = 0;

  uint32_t num_domains_ = 1;
  uint32_t build_domain_ = 0;
  bool build_launch_kept_ = false;
  bool launch_resetting_ = false;
  static inline thread_local uint32_t tl_exec_domain_ SIMX_TLS_IE = HOST_DOMAIN;
  static inline thread_local SimObjectBase* tl_exec_object_ SIMX_TLS_IE = nullptr;
  static inline thread_local uint64_t tl_cycles_ SIMX_TLS_IE = 0;
  bool audit_enabled_ = false;
  bool audit_requested_ = false;
  std::map<std::string, uint64_t> audit_hist_;

  // Deferred cross-domain call: runs in the target domain at the start of
  // its next cycle, in canonical order (enqueue cycle, then enqueue order).
  // Framework-internal — model code reaches it only through the channel and
  // event-link primitives; host loops test quiescence through idle().
  using CrossFn = SmallFunction<void(), 160>;

  // Cross-domain mailboxes. Each (target, source) pair owns a lane, so a
  // source appends without locks and the target drains alone. A lane is
  // triple-buffered by epoch (a cycle): during epoch e a source writes
  // epoch e (its tick) and e+1 (its delivery phase), while the target drains
  // epoch e-1 after the barrier that closed it. An entry written at cycle t
  // is due at t+1, which is always in the epoch after the one it was written
  // in, so draining epoch e-1 at the start of epoch e reproduces the
  // (due, source, order) delivery of a single shared inbox exactly.
  static constexpr uint32_t kCrossEpochs = 3;
  static constexpr uint32_t kHostLane = 0;   // lane 1 + d belongs to domain d

  struct CrossEntry {
    uint64_t due;
    CrossFn  fn;
  };
  struct alignas(64) CrossLane {
    std::vector<CrossEntry> buf[kCrossEpochs];
  };
  // mail_[target][lane]
  std::vector<std::vector<std::unique_ptr<CrossLane>>> mail_;

  static uint64_t cross_epoch(uint64_t cycle) {
    return cycle;
  }

  // Undelivered cross-domain calls (host context only: workers parked).
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

  // Host-context calls (tl_exec_domain_ == HOST_DOMAIN) must come from one
  // thread at a time; they share the host lane.
  void cross_call(uint32_t target_domain, CrossFn&& fn) {
    assert(target_domain < num_domains_);
    uint32_t src = tl_exec_domain_;
    uint64_t now = this->cycles();
    if (sync_active_ && src != HOST_DOMAIN && !sync_edge_[target_domain * num_domains_ + src]) {
      this->sync_unknown_edge(src, target_domain);
    }
    auto& lane = this->cross_lane(target_domain, (src == HOST_DOMAIN) ? kHostLane : (src + 1));
    lane.buf[cross_epoch(now) % kCrossEpochs].push_back({now + 1, std::move(fn)});
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

  void materialize_cross_lanes() {
    for (uint32_t t = 0; t < num_domains_; ++t) {
      for (uint32_t l = 0; l <= num_domains_; ++l) {
        (void)this->cross_lane_grow(t, l);
      }
    }
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

  // Channel registry (elaboration-time only) for topology validation.
  std::vector<SimChannelBase*> channels_;
  bool topo_validated_ = false;

  void register_channel(SimChannelBase* ch) {
    channels_.push_back(ch);
  }

  void unregister_channel(SimChannelBase* ch) {
    auto it = std::find(channels_.begin(), channels_.end(), ch);
    if (it != channels_.end()) {
      *it = channels_.back();
      channels_.pop_back();
    }
  }

  void validate_topology();

  // Run the calls due this epoch for one domain: the previous epoch's buffer
  // of each lane, in canonical order — the order that makes serial and
  // parallel execution agree.
  void drain_cross_calls_for(uint32_t d) {
    if (d >= mail_.size()) {
      return;
    }
    uint32_t nlanes = uint32_t(mail_[d].size());
    uint32_t idx = uint32_t((cross_epoch(this->cycles()) + kCrossEpochs - 1) % kCrossEpochs);
    bool any = false;
    for (auto& lane : mail_[d]) {
      if (lane && !lane->buf[idx].empty()) {
        any = true;
        break;
      }
    }
    if (!any) {
      return;
    }
    // One epoch is one cycle: every entry has the same due cycle, so the
    // canonical order is source order, host last.
    for (uint32_t l = 1; l <= nlanes; ++l) {
      this->run_lane(d, l % nlanes, idx);
    }
  }

  // A call may create a lane in serial execution, which can reallocate
  // mail_: index it afresh per lane. The lanes themselves never move.
  void run_lane(uint32_t d, uint32_t l, uint32_t idx) {
    auto* lane = mail_[d][l].get();
    if (lane == nullptr) {
      return;
    }
    auto& buf = lane->buf[idx];
    for (auto& e : buf) {
      e.fn();
    }
    buf.clear();
  }

  // ── Multi-threaded executor ────────────────────────────────────────────
  // One persistent worker per extra thread; domains are assigned round-robin
  // (the coordinator, thread 0, keeps stripe 0). A cycle is: drain the
  // mailboxes and tick each domain, then deliver each domain's due wheel
  // events, then one barrier. All per-domain state, the inflight count
  // included, is touched only by its owner; the only shared structures are
  // the mailbox lanes (one writer and one reader each, kept apart by epoch)
  // and the barrier itself.
  //
  // Inside run_until() the cycle-end barrier also starts the next cycle: its
  // release step commits the cycle and runs the host's stop check while every
  // worker is parked, so a cycle costs one barrier instead of two.
  uint32_t mt_threads_ = 1;
  uint32_t req_workers_ = 1;
  uint32_t mt_max_threads_ = std::numeric_limits<uint32_t>::max();
  bool mt_init_done_ = false;
  bool workers_started_ = false;
  std::vector<std::thread> workers_;
  std::vector<std::vector<Domain*>> stripe_domains_;
  std::atomic<uint64_t> barrier_epoch_{0};
  std::atomic<bool>     workers_exit_{false};

  // Each worker announces its arrival on its own cache line and the
  // coordinator collects them: no shared read-modify-write, so an arrival
  // does not contend with its peers' arrivals.
  struct alignas(64) ArriveFlag {
    std::atomic<uint64_t> epoch{0};
  };
  std::unique_ptr<ArriveFlag[]> arrive_;

  // Host stop check of the active run_until() loop (see cycle_end()).
  bool (*run_stop_)(void*) = nullptr;
  void* run_stop_arg_ = nullptr;
  bool run_active_ = false;

  // ── Neighbor synchronization (run_until with voters) ──────────────────
  // Domain d publishes the cycles it has completed and its vote, and starts
  // cycle c once every neighbor has completed c, so neighbors
  // stay within one cycle and the triple-buffered mailboxes stay valid.
  // Thread 0 owns the uncore domain, which neighbors every domain, and
  // decides the stop check for the cycle ending at c when it starts the
  // next one. A thread with a busy domain may run one cycle ahead, which
  // rules out a stop at c; a thread whose domains all voted idle waits for
  // the decision, so an all-idle check runs with every other thread parked.
  struct alignas(64) SyncProgress {
    std::atomic<uint64_t> word{0};   // completed cycles << 1 | busy vote
  };
  std::unique_ptr<SyncProgress[]> sync_progress_;
  std::vector<std::vector<uint32_t>> sync_deps_;   // neighbors per domain
  std::vector<uint8_t> sync_edge_;                 // [target * num_domains_ + source]
  alignas(64) std::atomic<uint64_t> sync_decided_{0};   // last cycle decided to continue
  std::atomic<bool> sync_stop_{false};
  bool sync_active_ = false;
  uint64_t sync_start_ = 0;
  bool (*sync_stop_fn_)(void*, bool) = nullptr;
  bool (*sync_vote_fn_)(void*, uint32_t) = nullptr;
  void* sync_stop_arg_ = nullptr;
  void* sync_vote_arg_ = nullptr;

  // Neighbors: domains joined by a channel bind chain, and the uncore with
  // every domain. A cross-domain call along any other pair aborts the run
  // (cross_call).
  void sync_init_graph() {
    uint32_t n = num_domains_;
    sync_edge_.assign(size_t(n) * n, 0);
    auto link = [&](uint32_t a, uint32_t b) {
      if (a != b) {
        sync_edge_[size_t(a) * n + b] = 1;
        sync_edge_[size_t(b) * n + a] = 1;
      }
    };
    for (auto* ch : channels_) {
      if (ch->source_ == nullptr && ch->sink_ != nullptr) {
        link(ch->module_->domain(), ch->endpoint()->module_->domain());
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

  __attribute__((noinline, noreturn)) void sync_unknown_edge(uint32_t src, uint32_t target) {
    std::cerr << "error: cross-domain call from domain " << src << " to domain " << target
              << ", which share no channel; neighbor synchronization cannot order it" << std::endl;
    std::abort();
  }

  // Thread w's part of a synchronized run_until().
  void sync_loop(uint32_t w) {
    const uint64_t step = 1;
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
            busy |= ((word >> 1) > c) || (word & 1);   // only a thread with a busy domain runs ahead
          }
          cycles_ = c;
          tl_exec_domain_ = HOST_DOMAIN;
          tl_exec_object_ = nullptr;
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
        tl_cycles_ = c;
        this->tick_domain(*dom);
        tl_cycles_ = c + 1;
        this->fire_bucket(*dom, c + 1);
        bool vote = sync_vote_fn_(sync_vote_arg_, dom->id);
        any_busy |= vote;
        sync_progress_[dom->id].word.store(((c + step) << 1) | (vote ? 1 : 0), std::memory_order_release);
      }
      tl_exec_domain_ = HOST_DOMAIN;
      tl_exec_object_ = nullptr;
      if (w != 0) {
        if (!any_busy) {
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
  }

  static constexpr uint32_t kBarrierSpins = 4096;

  // Spin briefly (the common case: peers arrive within microseconds), then
  // yield so a waiting thread does not starve a descheduled peer on an
  // oversubscribed host.
  template <typename Cond>
  static void spin_until(Cond&& cond) {
    uint32_t spins = 0;
    while (!cond()) {
      if (++spins < kBarrierSpins) {
      #if defined(__x86_64__)
        __builtin_ia32_pause();
      #endif
      } else {
        std::this_thread::yield();
      }
    }
  }

  // Thread `self` arrives (0 is the coordinator: whichever thread drives the
  // platform). The coordinator waits for every worker, runs on_release and
  // publishes the new epoch, so the release step's writes are visible to
  // every thread on the other side of the barrier.
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

  void barrier_wait(uint32_t self) {
    this->barrier_wait(self, []() {});
  }

  // One cycle of the domains in stripe w: tick, then deliver the events due
  // next cycle. A domain's wheel is written only by its own context (cross
  // traffic goes through the mailboxes), so the two phases need no barrier
  // between them; each thread advances its thread-local cycle view instead.
  void run_stripe(uint32_t w) {
    uint64_t cycle = cycles_;
    tl_cycles_ = cycle;
    for (auto* dom : stripe_domains_[w]) {
      this->tick_domain(*dom);
    }
    tl_cycles_ = cycle + 1;
    for (auto* dom : stripe_domains_[w]) {
      this->fire_bucket(*dom, cycle + 1);
    }
    tl_exec_domain_ = HOST_DOMAIN;
    tl_exec_object_ = nullptr;
  }

  // Release step of the cycle-end barrier: commit the cycle and, inside a
  // run_until() loop, evaluate the host's stop check.
  void cycle_end() {
    ++cycles_;
    if (run_active_ && run_stop_(run_stop_arg_)) {
      run_active_ = false;
    }
  }

  void fire_bucket(Domain& dom, uint64_t cycle) {
    dom.now = cycle;
    auto& bucket = dom.wheel[cycle & WHEEL_MASK];
    auto& lane = dom.next[cycle & 1];
    if (bucket.empty() && lane.empty()) {
      return;
    }
    // Delivery callbacks execute in the endpoint's domain context.
    tl_exec_domain_ = dom.id;
    tl_exec_object_ = nullptr;
    for (auto it = bucket.begin(); it != bucket.end();) {
      auto evt = &*it;
      if (evt->cycles() <= cycle) {
        evt->fire();
        it = bucket.erase(it);
        delete evt;
      } else {
        ++it; // future wraparound collision in this wheel slot
      }
    }
    // A delivery sends a delay-1 event to the other lane, so this one does
    // not grow while it fires.
    for (auto& e : lane) {
      e.fn(e, true);
    }
    lane.clear();
  }

  void init_mt(uint32_t num_workers) {
    mt_init_done_ = true;
    uint32_t count = std::min({std::max(1u, num_workers), num_domains_, mt_max_threads_});
    if (count < 2) {
      return;
    }
    audit_enabled_ = false;   // the audit histogram is not thread-safe
    mt_threads_ = count;
    // Materialize every domain and mailbox lane up front: workers must never resize
    // these containers or construct a slot concurrently.
    (void)this->domain_at(num_domains_ - 1);
    this->materialize_cross_lanes();
    // Round-robin, with the last partial round placed from the highest
    // thread down: thread 0 owns the uncore domain and runs the release
    // steps, so it takes no extra socket.
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
          this->barrier_wait(w);                         // parked between runs
          if (workers_exit_.load(std::memory_order_acquire)) {
            break;
          }
          if (sync_active_) {
            this->sync_loop(w);
            this->barrier_wait(w);                       // join
            continue;
          }
          // One cycle for tick(); cycles until the stop check fires for
          // run_until(). run_active_ is written only in a release step, so
          // every thread reads the same value after each barrier.
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
    this->barrier_wait(0);      // release parked workers into the exit check
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

  template <typename U> friend class SimChannel;
  template <typename U> friend class SimEventLink;
  template <typename U> friend class RegSlice;
  friend class SimChannelBase;
};

///////////////////////////////////////////////////////////////////////////////
// SimChannel Implementation
///////////////////////////////////////////////////////////////////////////////

// The storage of a channel endpoint: a delay line. Each slot holds a packet
// and the cycle it becomes visible, in ready order (equal cycles in send
// order), so a send writes its packet here at once and no delivery event is
// needed. Capacity is fixed; packets are constructed in place.
template <typename Pkt>
class DelayRing {
public:
  DelayRing() = default;

  explicit DelayRing(uint32_t capacity)
    : capacity_(capacity)
    , slots_(new Slot[capacity]) {
    __assert(capacity > 0, "DelayRing: capacity must be > 0");
  }

  DelayRing(const DelayRing&) = delete;
  DelayRing& operator=(const DelayRing&) = delete;

  DelayRing(DelayRing&& other) noexcept { this->take(other); }

  DelayRing& operator=(DelayRing&& other) noexcept {
    if (this != &other) {
      this->clear();
      this->take(other);
    }
    return *this;
  }

  ~DelayRing() { this->clear(); }

  uint32_t size() const { return size_; }
  uint32_t capacity() const { return capacity_; }

  // Ready cycle of the oldest packet; all ones when empty.
  uint64_t head_ready() const { return head_ready_; }

  const Pkt& front() const { return *slots_[head_].pkt(); }

  void pop() {
    slots_[head_].pkt()->~Pkt();
    head_ = this->wrap(head_ + 1);
    --size_;
    head_ready_ = (size_ != 0) ? slots_[head_].ready : ~uint64_t(0);
  }

  template <typename P>
  void insert(P&& pkt, uint64_t ready) {
    __assert(size_ < capacity_, "channel delay line overflow");
    uint32_t i = size_;
    uint32_t pos = this->wrap(head_ + i);
    while (i > 0) {
      uint32_t prev = this->wrap(head_ + i - 1);
      if (slots_[prev].ready <= ready) {
        break;
      }
      new (slots_[pos].buf) Pkt(std::move(*slots_[prev].pkt()));
      slots_[pos].ready = slots_[prev].ready;
      slots_[prev].pkt()->~Pkt();
      pos = prev;
      --i;
    }
    new (slots_[pos].buf) Pkt(std::forward<P>(pkt));
    slots_[pos].ready = ready;
    ++size_;
    if (i == 0) {
      head_ready_ = ready;
    }
  }

  void clear() {
    while (size_ != 0) {
      this->pop();
    }
  }

private:
  struct Slot {
    uint64_t ready;
    alignas(Pkt) unsigned char buf[sizeof(Pkt)];
    Pkt* pkt() { return std::launder(reinterpret_cast<Pkt*>(buf)); }
    const Pkt* pkt() const { return std::launder(reinterpret_cast<const Pkt*>(buf)); }
  };

  uint32_t wrap(uint32_t i) const { return (i >= capacity_) ? i - capacity_ : i; }

  void take(DelayRing& other) {
    capacity_ = other.capacity_;
    slots_ = std::move(other.slots_);
    head_ = other.head_;
    size_ = other.size_;
    head_ready_ = other.head_ready_;
    other.capacity_ = 0;
    other.head_ = other.size_ = 0;
    other.head_ready_ = ~uint64_t(0);
  }

  uint32_t capacity_ = 0;
  std::unique_ptr<Slot[]> slots_;
  uint32_t head_ = 0;
  uint32_t size_ = 0;
  uint64_t head_ready_ = ~uint64_t(0);
};

template <typename Pkt>
class SimChannel : public SimChannelBase {
public:
  static_assert(std::is_copy_constructible_v<Pkt>, "Packet must be copy constructible");
  using TxCallback = SmallFunction<void(const Pkt&, uint64_t), 48>;

  SimChannel(SimObjectBase* module, uint32_t capacity = 2)
    : SimChannelBase(module)
    , storage_(capacity)
    , pending_count_(0) {}

  SimChannel(const SimChannel& other)
    : SimChannelBase(other.module_)
    , storage_(other.storage_.capacity())
    , pending_count_(0)
    , reg_ready_(other.reg_ready_)
    , convert_fn_(other.convert_fn_)
    , deliver_fn_(other.deliver_fn_)
    , tx_cb_(other.tx_cb_) {
    sink_ = nullptr;
    source_ = nullptr;
    has_tx_cb_ = other.has_tx_cb_;
  }

  SimChannel& operator=(const SimChannel& other) {
    if (this != &other) {
      this->module_ = other.module_;
      sink_ = nullptr;
      source_ = nullptr;
      endpoint_cache_ = nullptr;
      this->resolve_domain();
      pop_cb_ = {};
      push_waker_ = nullptr;
      pop_waker_ = nullptr;
      storage_ = DelayRing<Pkt>(other.storage_.capacity());
      pending_count_ = 0;
      reg_ready_ = other.reg_ready_;
      freed_cycle_ = ~uint64_t(0);
      freed_count_ = 0;
      convert_fn_ = other.convert_fn_;
      deliver_fn_ = other.deliver_fn_;
      tx_cb_ = other.tx_cb_;
      has_tx_cb_ = other.has_tx_cb_;
      delivery_mode_ = 0;
    }
    return *this;
  }

  void bind(SimChannel<Pkt>* sink) {
    this->bind_setup(sink);
    convert_fn_.reset();
    deliver_fn_.reset();
  }

  template <typename U, typename Converter>
  void bind(SimChannel<U>* sink, Converter&& converter) {
    static_assert(std::is_invocable_r_v<U, Converter, Pkt>, "Converter signature mismatch");
    this->bind_setup(sink);
    deliver_fn_ = [sink, conv = converter](const Pkt& pkt, uint64_t ready) {
      sink->deliver(conv(pkt), ready);
    };
    convert_fn_ = [sink, conv = std::forward<Converter>(converter)](const Pkt& pkt) {
      sink->receive_packet(conv(pkt));
    };
  }

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<Pkt, U>>>
  void bind(SimChannel<U>* sink) {
    this->bind_setup(sink);
    deliver_fn_ = [sink](const Pkt& pkt, uint64_t ready) {
      sink->deliver(U(pkt), ready);
    };
    convert_fn_ = [sink](const Pkt& pkt) {
      sink->receive_packet(pkt);
    };
  }

  // Fires on delivery, in the endpoint domain's execution context: a
  // callback on a cross-domain chain must only touch endpoint-side state.
  template <typename F>
  void tx_callback(F&& callback) {
    tx_cb_ = std::forward<F>(callback);
    has_tx_cb_ = true;
  }

  bool full() const override {
    if (sink_) {
      return this->endpoint()->full();
    }
    assert(!SimPlatform::instance().mt_cross_access(module_));
    return this->occupancy() + this->freed_now() >= storage_.capacity();
  }

  // A registered ready: a slot freed by a pop is offered to producers from the
  // next cycle, as a buffer whose full flag is a register does, whatever order
  // the producer and the consumer tick in.
  void set_registered_ready(bool enable) { reg_ready_ = enable; }

  void send(const Pkt& pkt, uint64_t delay = 1) {
    auto& platform = SimPlatform::instance();
    // A send whose endpoint lives in another domain is handed to that
    // domain's owner as a deferred call: it re-issues the send locally next
    // cycle with one less delay, landing on the same absolute cycle. All
    // channel state is thereby touched only by its owning thread. Capacity
    // needs no check here — cross-domain producers are credit-gated at the
    // boundary, so a deferred send always finds space.
    if (platform.is_cross_exec(this->endpoint()->module_)) {
      __assert(delay >= 1, "cross-domain send requires a registered delay");
      this->cross_send(Pkt(pkt), delay);
      return;
    }
    __assert(!this->full(), "channel is full");
    if (platform.audit_enabled()) {
      platform.audit_send(this, delay);
    }
    if (this->direct_delivery()) {
      this->deliver(pkt, this->endpoint()->clock() + delay);
      return;
    }
    this->reserve();
    platform.schedule(this, pkt, delay);
  }

  void send(Pkt&& pkt, uint64_t delay = 1) {
    auto& platform = SimPlatform::instance();
    if (platform.is_cross_exec(this->endpoint()->module_)) {
      __assert(delay >= 1, "cross-domain send requires a registered delay");
      this->cross_send(std::move(pkt), delay);
      return;
    }
    __assert(!this->full(), "channel is full");
    if (platform.audit_enabled()) {
      platform.audit_send(this, delay);
    }
    if (this->direct_delivery()) {
      this->deliver(std::move(pkt), this->endpoint()->clock() + delay);
      return;
    }
    this->reserve();
    platform.schedule(this, std::move(pkt), delay);
  }

  [[nodiscard]] bool try_send(const Pkt& pkt, uint64_t delay = 1) {
    if (this->full()) return false;
    this->send(pkt, delay);
    return true;
  }

  [[nodiscard]] bool try_send(Pkt&& pkt, uint64_t delay = 1) {
    if (this->full()) return false;
    this->send(std::move(pkt), delay);
    return true;
  }

  // Hand a send to the endpoint's domain: it re-issues the send locally with
  // one less delay, landing on the same absolute cycle. A packet that fits
  // the call's inline buffer rides by value; a larger one is boxed.
  void cross_send(Pkt&& pkt, uint64_t delay) {
    auto& platform = SimPlatform::instance();
    uint32_t target = this->endpoint()->module_->domain();
    if constexpr (sizeof(Pkt) <= kInlineCrossPkt) {
      platform.cross_call(target, [this, p = std::move(pkt), delay]() {
        this->send(p, delay - 1);
      });
    } else {
      platform.cross_call(target, [this, p = std::make_shared<Pkt>(std::move(pkt)), delay]() {
        this->send(*p, delay - 1);
      });
    }
  }

  static constexpr size_t kInlineCrossPkt = 128;

  bool empty() const override {
    this->assert_endpoint();
    return this->queue_empty();
  }

  const Pkt& peek() const {
    this->assert_endpoint();
    __assert(!this->queue_empty(), "channel is empty");
    return this->queue_front();
  }

  void pop() {
    this->assert_endpoint();
    __assert(!this->queue_empty(), "channel is empty");
    this->queue_pop();
  }

  [[nodiscard]] bool try_pop(Pkt* out) {
    __assert(out != nullptr, "output target is null");
    if (this->empty()) return false;
    *out = this->peek();
    this->pop();
    return true;
  }

  uint32_t size() const override {
    if (sink_) {
      return this->endpoint()->size();
    }
    assert(!SimPlatform::instance().mt_cross_access(module_));
    return this->occupancy();
  }

  uint32_t capacity() const override {
    if (sink_) {
      return this->endpoint()->capacity();
    }
    return storage_.capacity();
  }

protected:
  void reserve() override {
    if (sink_) {
      this->endpoint()->reserve();
    } else {
      ++pending_count_;
      ++this->inflight();
      wake_module(module_);
      if (push_waker_) {
        wake_module(push_waker_);
      }
    }
  }

  void receive_packet(const Pkt& pkt) {
    if (tx_cb_) {
      tx_cb_(pkt, SimPlatform::instance().cycles());
    }
    if (sink_) {
      if (convert_fn_) {
        convert_fn_(pkt);
      } else {
        auto* sink = static_cast<SimChannel<Pkt>*>(sink_);
        sink->receive_packet(pkt);
      }
      return;
    }
    __assert(pending_count_ > 0, "pending count underflow");
    --pending_count_;
    storage_.insert(pkt, this->clock());
  }

  // Direct delivery: down the chain (converting where a hop converts) into
  // the endpoint's delay line, visible at `ready`. The endpoint's owner wakes
  // now, at the send, as a reservation does.
  template <typename P>
  void deliver(P&& pkt, uint64_t ready) {
    if (sink_) {
      if (deliver_fn_) {
        deliver_fn_(pkt, ready);
      } else {
        static_cast<SimChannel<Pkt>*>(sink_)->deliver(std::forward<P>(pkt), ready);
      }
      return;
    }
    ++this->inflight();
    storage_.insert(std::forward<P>(pkt), ready);
    wake_module(module_);
    if (push_waker_) {
      wake_module(push_waker_);
    }
  }

private:
  bool forwarded() const { return sink_ != nullptr; }

  bool direct_delivery() {
    if (__builtin_expect(delivery_mode_ == 0, 0)) {
      delivery_mode_ = this->chain_has_tx_cb() ? 2 : 1;
    }
    return delivery_mode_ == 1;
  }

  void bind_setup(SimChannelBase* sink) {
    __assert(sink != nullptr, "bind target is null");
    __assert(sink_ == nullptr, "channel already bound");
    sink_ = sink;
    sink->source_ = this;
  }

  void assert_endpoint() const {
    __assert(!forwarded(), "cannot read from a forwarded channel");
  }

  uint32_t occupancy() const { return storage_.size() + pending_count_; }

  // No packet ready at the endpoint domain's current cycle.
  bool queue_empty() const {
    return storage_.head_ready() > this->clock();
  }
  uint32_t queue_size() const { return storage_.size(); }
  const Pkt& queue_front() const { return storage_.front(); }
  uint32_t freed_now() const {
    auto& platform = SimPlatform::instance();
    return (reg_ready_ && freed_cycle_ == platform.cycles() && freed_gen_ == platform.reset_gen())
         ? freed_count_ : 0;
  }
  void queue_pop() {
    if (reg_ready_) {
      auto& platform = SimPlatform::instance();
      if (freed_cycle_ != platform.cycles() || freed_gen_ != platform.reset_gen()) {
        freed_cycle_ = platform.cycles();
        freed_gen_ = platform.reset_gen();
        freed_count_ = 0;
      }
      ++freed_count_;
    }
    storage_.pop();
    --this->inflight();
    if (pop_waker_) {
      wake_module(pop_waker_);
    }
    if (pop_cb_) {
      pop_cb_();
    }
  }
  DelayRing<Pkt> storage_;
  uint32_t pending_count_;
  bool reg_ready_ = false;
  uint64_t freed_cycle_ = ~uint64_t(0);
  uint64_t freed_gen_ = 0;
  uint32_t freed_count_ = 0;
  SmallFunction<void(const Pkt&), 48> convert_fn_;
  SmallFunction<void(const Pkt&, uint64_t), 48> deliver_fn_;
  TxCallback tx_cb_;

  template <typename U> friend class SimChannel;
  template <typename U> friend class SimChannelEvent;
  friend class SimChannelBase;
  friend class SimPlatform;
};

// Constructs a std::array of N SimChannel<Pkt> instances, each bound to owner.
// SimChannel is not default-constructible, so a plain brace-init list cannot be used.
//     std::array<SimChannel<Pkt>, N> Inputs = make_sim_channels<Pkt, N>(this);

namespace detail {
template <typename Pkt, std::size_t N, std::size_t... Is>
inline std::array<SimChannel<Pkt>, N> make_sim_channels_impl(SimObjectBase* owner, uint32_t capacity, std::index_sequence<Is...>) {
  return std::array<SimChannel<Pkt>, N>{ ((void)Is, SimChannel<Pkt>(owner, capacity))... };
}
} // namespace detail

template <typename Pkt, std::size_t N>
inline std::array<SimChannel<Pkt>, N> make_sim_channels(SimObjectBase* owner, uint32_t capacity = 2) {
  return detail::make_sim_channels_impl<Pkt, N>(owner, capacity, std::make_index_sequence<N>{});
}

///////////////////////////////////////////////////////////////////////////////
// SimEventLink
///////////////////////////////////////////////////////////////////////////////

// One-way, typed event link for control-plane strobes (doorbells, barrier
// arrive/resume, completion kicks). Both ends are declared members: the
// receiver binds a member-function handler to its end, and the sender's end
// is wired to the receiver's at elaboration with the same bind() idiom
// channels use (fan-in is allowed: multiple out-ends may target one handler
// end). Delivery is non-refusable and carries a registered >= 1 cycle
// latency: the bound handler runs in the receiving module's execution
// context at cycle C + delay, with same-cycle deliveries merged in canonical
// (due, src, seq) order — identical serial and parallel. Unlike a channel, a
// link has no queue, no occupancy, and no back-pressure; it is the only way
// to trigger behavior on a module in another execution domain.
template <typename Msg>
class SimEventLink {
public:
  static_assert(std::is_copy_constructible_v<Msg>, "Message must be copy constructible");

  SimEventLink(SimObjectBase* module) : module_(module) {}

  // Copying is only for container fill at construction: bindings do not
  // carry over.
  SimEventLink(const SimEventLink& other) : module_(other.module_) {}

  SimEventLink& operator=(const SimEventLink& other) {
    if (this != &other) {
      module_ = other.module_;
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

  // Receiver end: bind the member-function handler invoked on delivery.
  template <typename Impl>
  void bind(Impl* obj, void (Impl::*handler)(const Msg&)) {
    __assert(obj != nullptr, "handler object is null");
    __assert(sink_ == nullptr && !handler_, "link end already bound");
    handler_ = [obj, handler](const Msg& msg) { (obj->*handler)(msg); };
  }

  // Fire-and-forget delivery: the endpoint handler runs in its module's
  // context at cycle C + delay. Cannot fail and cannot be refused.
  void send(const Msg& msg, uint64_t delay = 1) {
    __assert(delay >= 1, "event link delivery requires a registered delay");
    auto* ep = this->endpoint();
    __assert(!!ep->handler_, "event link has no bound handler");
    SimPlatform::instance().cross_call(ep->module_->domain(),
      [ep, msg, delay]() {
        if (delay > 1) {
          ep->send(msg, delay - 1);  // re-issued from the endpoint's own context
        } else {
          ep->handler_(msg);
        }
      });
  }

private:
  SimEventLink<Msg>* endpoint() {
    auto* e = this;
    while (e->sink_) {
      e = e->sink_;
    }
    return e;
  }

  SimObjectBase* module_;
  SimEventLink<Msg>* sink_ = nullptr;
  SmallFunction<void(const Msg&), 48> handler_;
};

///////////////////////////////////////////////////////////////////////////////
// Object Creation & Platform Implementation
///////////////////////////////////////////////////////////////////////////////

// Detects whether on_tick()/on_reset() are public on T via SFINAE.
// create_object<Impl> static_asserts these are not public.
namespace detail {
  template <typename T>
  auto detect_on_tick_public(int)
      -> decltype(std::declval<T&>().on_tick(), std::true_type{});
  template <typename T>
  std::false_type detect_on_tick_public(...);

  template <typename T>
  auto detect_on_reset_public(int)
      -> decltype(std::declval<T&>().on_reset(), std::true_type{});
  template <typename T>
  std::false_type detect_on_reset_public(...);
}

template <typename T>
struct is_on_tick_public  : decltype(detail::detect_on_tick_public<T>(0))  {};
template <typename T>
struct is_on_reset_public : decltype(detail::detect_on_reset_public<T>(0)) {};

template <typename Impl>
class SimObject : public SimObjectBase {
public:
  using Ptr = std::shared_ptr<Impl>;

  template <typename... Args>
  static Ptr Create(Args&&... args) {
    return SimPlatform::instance().create_object<Impl>(std::forward<Args>(args)...);
  }

protected:
  SimObject(const SimContext& ctx, const std::string& name)
    : SimObjectBase(ctx, name) {}

  // Lifecycle callbacks — must remain protected. Each derivative must declare
  // `friend class SimObject<Self>` for override-detection to work.
  void on_tick()  {}
  void on_reset() {}

private:
  Impl* impl() { return static_cast<Impl*>(this); }
  void do_reset() override { impl()->on_reset(); }
  void do_tick()  override { impl()->on_tick();  }

  // Returns true if Impl overrides on_tick()/on_reset() rather than inheriting
  // the base no-op. Uses member-pointer comparison; must be a static member of
  // SimObject<Impl> to access the protected member pointer across boundaries.
  template <typename T = Impl>
  static bool has_own_tick() {
    using F = void (T::*)();
    return static_cast<F>(&T::on_tick) != static_cast<F>(&SimObject<Impl>::on_tick);
  }

  template <typename T = Impl>
  static bool has_own_reset() {
    using F = void (T::*)();
    return static_cast<F>(&T::on_reset) != static_cast<F>(&SimObject<Impl>::on_reset);
  }

  friend class SimPlatform;
};

// True only for direct SimObject<Impl> CRTP derivatives.
// Multi-level chains (Derived→Intermediate→SimObject<Intermediate>) yield false,
// so they are conservatively treated as always-active.
template <typename Impl, typename = void>
struct has_direct_simobject_base : std::false_type {};

template <typename Impl>
struct has_direct_simobject_base<Impl,
    std::void_t<decltype(static_cast<SimObject<Impl>*>(std::declval<Impl*>()))>>
  : std::true_type {};

template <typename Impl, typename... Args>
std::shared_ptr<Impl> SimPlatform::create_object(Args&&... args) {
  static_assert(!is_on_tick_public<Impl>::value,
      "on_tick() must be protected — only SimPlatform may call it");
  static_assert(!is_on_reset_public<Impl>::value,
      "on_reset() must be protected — only SimPlatform may call it");
  auto obj = std::make_shared<Impl>(SimContext{}, std::forward<Args>(args)...);
  objects_.push_back(obj);
  // Skip-inactive optimization applies only to direct CRTP derivatives; multi-level
  // chains are conservatively kept active because &Impl::on_tick is inaccessible there.
  if constexpr (has_direct_simobject_base<Impl>::value) {
    if (SimObject<Impl>::template has_own_tick<Impl>()) {
      this->add_scan(obj.get());
    }
    if (SimObject<Impl>::template has_own_reset<Impl>()) {
      active_reset_.push_back(obj.get());
    }
  } else {
    this->add_scan(obj.get());
    active_reset_.push_back(obj.get());
  }
  return obj;
}

template <typename Pkt>
void SimPlatform::schedule(SimChannel<Pkt>* channel, const Pkt& pkt, uint64_t delay) {
  auto& dom = domain_at(channel->endpoint()->module_->domain_);
  if (delay == 0) {
    // Same-cycle delivery is legal only within one domain (or from host
    // context between cycles) — a cross-domain edge must be registered.
    __assert(tl_exec_domain_ == HOST_DOMAIN
        || &dom == &domain_at(tl_exec_domain_), "cross-domain delta send");
    lane_push(dom.imm, channel, pkt);
  } else if (delay == 1) {
    lane_push(dom.next[(this->cycles() + 1) & 1], channel, pkt);
  } else {
    uint64_t fire_cycle = this->cycles() + delay;
    auto evt = new SimChannelEvent<Pkt>(channel, pkt, fire_cycle);
    dom.wheel[fire_cycle & WHEEL_MASK].push_back(evt);
  }
}
template <typename Pkt>
void SimPlatform::schedule(SimChannel<Pkt>* channel, Pkt&& pkt, uint64_t delay) {
  auto& dom = domain_at(channel->endpoint()->module_->domain_);
  if (delay == 0) {
    // Same-cycle delivery is legal only within one domain (or from host
    // context between cycles) — a cross-domain edge must be registered.
    __assert(tl_exec_domain_ == HOST_DOMAIN
        || &dom == &domain_at(tl_exec_domain_), "cross-domain delta send");
    lane_push(dom.imm, channel, std::move(pkt));
  } else if (delay == 1) {
    lane_push(dom.next[(this->cycles() + 1) & 1], channel, std::move(pkt));
  } else {
    uint64_t fire_cycle = this->cycles() + delay;
    auto evt = new SimChannelEvent<Pkt>(channel, std::move(pkt), fire_cycle);
    dom.wheel[fire_cycle & WHEEL_MASK].push_back(evt);
  }
}

template <typename Pkt, typename Func>
void SimPlatform::schedule(Func&& func, const Pkt& pkt, uint64_t delay, uint32_t domain) {
  __assert(delay != 0, "scheduled callbacks require a registered delay");
  auto& dom = domain_at(domain);
  uint64_t fire_cycle = this->cycles() + delay;
  auto evt = new SimCallEvent<Pkt>(std::forward<Func>(func), pkt, fire_cycle);
  if (delay == 1) {
    dom.next[fire_cycle & 1].push_back({&lane_fire_event, static_cast<SimEventBase*>(evt), {}});
  } else {
    dom.wheel[fire_cycle & WHEEL_MASK].push_back(evt);
  }
}

// One-time topology validation, run at the first reset after elaboration:
// every bound channel chain crossing an execution-domain boundary must pass
// through a registered boundary stage. An unregistered crossing is safe
// serially (sends marshal through the inboxes) but couples timing to the
// partition layout, so it is reported once and caps execution at one thread.
inline void SimPlatform::validate_topology() {
  if (topo_validated_) {
    return;
  }
  topo_validated_ = true;
  if (num_domains_ < 2) {
    return;
  }
  bool clamped = false;
  for (auto* ch : channels_) {
    if (ch->source_ != nullptr || ch->sink_ == nullptr) {
      continue; // interior link or standalone endpoint
    }
    auto* ep = ch->endpoint();
    if (ep->module_->domain() == ch->module_->domain()) {
      continue;
    }
    bool registered = false;
    for (auto* e = ch; e != nullptr; e = e->sink_) {
      if (e->boundary_stage_) {
        registered = true;
        break;
      }
    }
    if (!registered) {
      std::cout << "SIMX-MT: unregistered cross-domain edge: "
                << ch->module_->name() << " -> " << ep->module_->name()
                << std::endl;
      clamped = true;
    }
  }
  if (clamped) {
    std::cout << "SIMX-MT: parallel execution disabled until every "
                 "cross-domain edge carries a registered stage" << std::endl;
    mt_max_threads_ = 1;
  }
}

// A kernel launch resets every object but those built under a
// LaunchKeepScope: the hardware does not reset between a transfer and the
// launch that follows it, so the transfer's line fills may still be in
// flight in the kept memory side when the kernel starts. Their events stay
// scheduled and time keeps counting; the rest of the model is quiescent.
inline void SimPlatform::reset_launch() {
  launch_resetting_ = true;
  for (auto* object : active_reset_) {
    if (!object->launch_kept_) {
      object->do_reset();
    }
  }
  launch_resetting_ = false;
  this->wake_all();
}

inline void SimPlatform::reset() {
  this->validate_topology();

  // Clear any lingering events from the previous run. Dropping an
  // undelivered event does not unwind its reservation accounting, so
  // callers reset only from quiescence (see idle()).
  for (auto& dom : domains_) {
    for (auto& bucket : dom->wheel) {
      while (!bucket.empty()) {
        auto it = bucket.begin();
        auto evt = &*it;
        bucket.erase(it);
        delete evt;
      }
    }
    lane_drop(dom->imm);
    lane_drop(dom->next[0]);
    lane_drop(dom->next[1]);
  }

  // clear sim objects (only those that override reset())
  for (auto* object : active_reset_) {
    object->do_reset();
  }

  // Re-arm every tick-gated object for the new run.
  this->wake_all();

  // Drop undelivered cross-domain calls from the previous run. The lanes
  // themselves are kept: workers index them concurrently once started,
  // so the vector must never shrink after init_mt materialized it.
  this->clear_cross_lanes();

  // Reset timing
  cycles_ = 0;
  for (auto& dom : domains_) {
    dom->now = 0;
  }
  ++reset_gen_;
}

inline void SimPlatform::tick() {
  if (!mt_init_done_) {
    this->init_mt(req_workers_);
  }
  if (workers_started_) {
    // Parallel cycle: workers are parked on the entry barrier between
    // tick() calls, so everything outside this function is single-threaded.
    // Two barriers per cycle — release and join; the global counter is
    // committed in the join barrier's release step, and each thread's
    // per-phase cycle view is thread-local (see cycles()).
    this->barrier_wait(0);                         // release workers
    this->run_stripe(0);
    this->barrier_wait(0, [this]() { this->cycle_end(); });   // join; workers park
    return;
  }

  // Serial reference path: tick each domain — its due cross calls, immediate
  // events, then its scan list in creation order (the order fixes the
  // delta-event interleaving within the domain). Domains only interact
  // through registered (latency >= 1) edges, so the domain execution order
  // within a cycle is not observable.
  tl_cycles_ = cycles_;
  for (auto& dom : domains_) {
    this->tick_domain(*dom);
  }
  ++cycles_;
  tl_cycles_ = cycles_;

  // Deliver each domain's registered events due this cycle.
  for (auto& dom : domains_) {
    this->fire_bucket(*dom, cycles_);
  }
  tl_exec_domain_ = HOST_DOMAIN;
  tl_exec_object_ = nullptr;
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
  this->barrier_wait(0, [this]() { run_active_ = true; });   // release workers
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
  this->barrier_wait(0);                           // release workers
  this->sync_loop(0);
  this->barrier_wait(0);                           // join: workers parked
  sync_active_ = false;
  tl_cycles_ = cycles_;
  sync_stop_fn_ = nullptr;
  sync_vote_fn_ = nullptr;
  sync_stop_arg_ = nullptr;
  sync_vote_arg_ = nullptr;
}

inline void SimPlatform::tick_domain(Domain& dom) {
  dom.now = tl_cycles_;
  tl_exec_domain_ = dom.id;
  tl_exec_object_ = nullptr;
  // Start-of-cycle for this domain: integrate deferred cross-domain work.
  {
    this->drain_cross_calls_for(dom.id);
  }
  if (!dom.imm.empty()) {
    fire_immediate_events(dom);
  }
  // Every object on a domain's scan list belongs to that domain, so the
  // execution domain set above holds for the whole scan. The current object
  // is tracked only for the boundary audit, its sole reader.
  // Visit the ready set in creation order. The word is re-read after each
  // tick, so an object woken by one ahead of it ticks this cycle and one
  // woken by an object behind it ticks next cycle, as in a scan of every
  // object that tests a flag.
  const bool track_object = audit_enabled_;
  uint64_t* words = dom.active_words ? dom.active() : nullptr;
  SimObjectBase* const* objects = dom.scan.data();
  for (uint32_t w = 0, nw = dom.active_words; w < nw; ++w) {
    uint64_t bits = words[w];
    while (bits != 0) {
      uint32_t b = uint32_t(__builtin_ctzll(bits));
      auto* object = objects[size_t(w) * 64 + b];
      if (track_object) {
        tl_exec_object_ = object;
      }
      object->do_tick();
      if (!dom.imm.empty()) {
        fire_immediate_events(dom);
      }
      bits = words[w] & (~uint64_t(1) << b);
    }
  }
}

inline void SimPlatform::cleanup() {
  this->stop_workers();
  if (audit_enabled_ && !audit_hist_.empty()) {
    this->dump_audit(std::cout);
    audit_hist_.clear();
  }
  active_reset_.clear();
  objects_.clear();

  for (auto& dom : domains_) {
    for (auto& bucket : dom->wheel) {
      while (!bucket.empty()) {
        auto it = bucket.begin();
        auto evt = &*it;
        bucket.erase(it);
        delete evt;
      }
    }
    lane_drop(dom->imm);
    lane_drop(dom->next[0]);
    lane_drop(dom->next[1]);
  }
  domains_.clear();
  mail_.clear();
  sync_progress_.reset();
  sync_deps_.clear();
  sync_edge_.clear();
  num_domains_ = 1;
  cycles_ = 0;
  // Allow a subsequent initialize() to elaborate and parallelize afresh.
  mt_init_done_ = false;
  mt_max_threads_ = std::numeric_limits<uint32_t>::max();
  topo_validated_ = false;
  audit_enabled_ = audit_requested_;
}

inline void SimPlatform::fire_immediate_events(Domain& dom) {
  // Immediate events fire in schedule order, including those a firing
  // appends, which come after every queued one: the same order as draining
  // delta levels one by one. The lane may grow while it fires, so each
  // event is copied out before it runs.
  for (size_t i = 0; i < dom.imm.size(); ++i) {
    LaneEvent e = dom.imm[i];
    e.fn(e, true);
  }
  dom.imm.clear();
}

template <typename Pkt>
void SimChannelEvent<Pkt>::fire() {
  channel_->receive_packet(pkt_);
}

inline SimObjectBase::SimObjectBase(const SimContext&, const std::string& name)
    : name_(name)
    , domain_(SimPlatform::instance().build_domain())
    , launch_kept_(SimPlatform::instance().build_launch_kept()) {}

inline SimChannelBase::SimChannelBase(SimObjectBase* module)
    : module_(module)
    , sink_(nullptr)
    , source_(nullptr) {
  SimPlatform::instance().register_channel(this);
  this->resolve_domain();
}

inline int64_t& SimChannelBase::resolve_inflight() {
  return SimPlatform::instance().domain_at(module_->domain()).inflight;
}

inline const uint64_t& SimChannelBase::resolve_clock() {
  return SimPlatform::instance().domain_at(module_->domain()).now;
}

inline SimChannelBase::~SimChannelBase() {
  SimPlatform::instance().unregister_channel(this);
}

inline void SimPlatform::audit_send(SimChannelBase* channel, uint64_t delay) {
  auto* endpoint = channel;
  while (endpoint->sink_) {
    endpoint = endpoint->sink_;
  }
  uint32_t dst = endpoint->module_->domain_;
  if (tl_exec_domain_ == dst) {
    return;
  }
  std::stringstream key;
  if (tl_exec_domain_ == HOST_DOMAIN) {
    key << "host";
  } else {
    key << "d" << tl_exec_domain_ << ":"
        << (tl_exec_object_ ? tl_exec_object_->name() : "?");
  }
  key << " -> d" << dst << ":" << endpoint->module_->name()
      << " delay=" << delay;
  ++audit_hist_[key.str()];
}

inline void SimPlatform::dump_audit(std::ostream& os) {
  os << "BOUNDARY-AUDIT: " << num_domains_ << " domains, "
     << audit_hist_.size() << " distinct cross-domain edges" << std::endl;
  for (auto& [key, count] : audit_hist_) {
    os << "BOUNDARY-AUDIT: " << key << " count=" << count << std::endl;
  }
}

} // namespace vortex
