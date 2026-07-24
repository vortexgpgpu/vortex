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

// Functional simulation kernel: same executor as the timed kernel in
// simobject.h (domains, lockstep multi-threading, canonical cross-domain
// ordering) with timing removed — every delivery latency collapses to one
// cycle (delay 0 keeps its same-cycle delta ordering) and back-pressure is
// disabled (channels are unbounded and never refuse). Cycle counts are
// monotonic but non-physical: a functional build must never feed perf_gate,
// model_parity, or baseline regeneration. A build selects this kernel with
// -DSIMX_FUNCTIONAL (see sim/simx/types.h); the inline namespace makes a
// mixed timed/functional link fail at link time for kernel symbols.
// RegSlice lives in the outer namespace and is not covered by that guard —
// it relies on each build tree selecting one kernel uniformly.
namespace vortex {

// The registered boundary stage (sim/simx/regslice.h) lives in the outer
// namespace so one declaration serves both kernels.
template <typename T> class RegSlice;

inline namespace sim_functional {

// Forward declarations
class SimPlatform;
class SimObjectBase;
template <typename T> class SimChannel;
template <typename T> class SimEventLink;

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
  // hides work. Objects opt in by calling tick_sleep() from on_tick();
  // tick_wake() re-arms from non-channel entry points (e.g. a flush request).
  void tick_sleep() { tick_active_ = false; }
  void tick_wake()  { tick_active_ = true; }

private:
  std::string name_;
  bool tick_active_ = true;
  uint32_t domain_ = 0;
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

  virtual bool empty() const = 0;
  virtual bool full() const = 0;
  virtual uint32_t size() const = 0;
  virtual uint32_t capacity() const = 0;

  // Process-global count of packets that have been sent but not yet consumed.
  // Atomic: incremented/decremented from domain worker threads; only read for
  // the termination check at a barrier point.
  static std::atomic<uint64_t>& inflight_count() {
    static std::atomic<uint64_t> count{0};
    return count;
  }

protected:
  // Defined after SimPlatform: registers with the platform channel registry
  // (used for topology validation).
  explicit SimChannelBase(SimObjectBase* module);

  virtual void reserve() = 0;

  // Re-arm a tick-gated module (see SimObjectBase::tick_sleep).
  static void wake_module(SimObjectBase* module) {
    module->tick_active_ = true;
  }

  SimObjectBase*  module_;
  SimChannelBase* sink_;
  SimChannelBase* source_;
  SimChannelBase* endpoint_cache_ = nullptr;
  bool boundary_stage_ = false;
  SmallFunction<void(), 48> pop_cb_;

  // Terminal endpoint of the bind chain (cached: bindings are immutable
  // after construction).
  SimChannelBase* endpoint() {
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
  static SimPlatform& instance() {
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

  // Sends issued outside any object tick (driver/host calls).
  static constexpr uint32_t HOST_DOMAIN = 0xffffffffu;

  // True when the platform holds no undelivered work: no packet sent but not
  // yet consumed, and no deferred cross-domain delivery in flight. This is
  // the only valid quiescence test — host loops must not reassemble it from
  // framework internals.
  bool idle() const {
    return SimChannelBase::inflight_count() == 0 && this->cross_pending() == 0;
  }

  // Domain topology API — used only by topology containers (Socket/Cluster);
  // leaf modules inherit the scope in effect when they are created.
  uint32_t alloc_domain() { return num_domains_++; }
  uint32_t num_domains() const { return num_domains_; }
  uint32_t build_domain() const { return build_domain_; }

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
  void tick();

private:
  // Timing Wheel Configuration. With every delay collapsed to one cycle only
  // two buckets are ever populated, but the wheel is kept: its cost is a
  // masked index, and sharing the timed kernel's structure keeps the two
  // kernels line-for-line comparable.
  static constexpr uint64_t WHEEL_SIZE = 4096;
  static constexpr uint64_t WHEEL_MASK = WHEEL_SIZE - 1;

  // Per-domain kernel state: scan list, timing wheel, and immediate-event
  // list are private to one execution domain, so a domain can be ticked by
  // its own worker without touching another domain's structures. Events are
  // routed to the ENDPOINT's domain at schedule time.
  struct Domain {
    std::vector<SimObjectBase*> scan;   // creation order within the domain
    std::vector<LinkedList<SimEventBase, &SimEventBase::list_>> wheel;
    LinkedList<SimEventBase, &SimEventBase::list_> imm;
    uint32_t delta = 0;
    uint32_t id = 0;
    Domain() : wheel(WHEEL_SIZE) {}
  };

  Domain& domain_at(uint32_t d) {
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

  uint32_t num_domains_ = 1;
  uint32_t build_domain_ = 0;
  static inline thread_local uint32_t tl_exec_domain_ = HOST_DOMAIN;
  static inline thread_local SimObjectBase* tl_exec_object_ = nullptr;
  static inline thread_local uint64_t tl_cross_seq_ = 0;
  static inline thread_local uint64_t tl_cycles_ = 0;
  bool audit_enabled_ = false;
  bool audit_requested_ = false;
  std::map<std::string, uint64_t> audit_hist_;

  // Deferred cross-domain call: runs in the target domain at the start of
  // its next cycle, in canonical order (enqueue cycle, then enqueue order).
  // Framework-internal — model code reaches it only through the channel and
  // event-link primitives; host loops test quiescence through idle().
  using CrossFn = SmallFunction<void(), 160>;

  uint64_t cross_pending() const {
    return cross_pending_.load(std::memory_order_acquire);
  }

  // Host-context calls (tl_exec_domain_ == HOST_DOMAIN) must come from one
  // thread at a time — the per-thread sequence numbers that order same-cycle
  // entries are only comparable within a single source thread.
  void cross_call(uint32_t target_domain, CrossFn&& fn) {
    assert(target_domain < num_domains_);
    auto& inbox = this->cross_inbox_at(target_domain);
    uint32_t src = tl_exec_domain_;
    uint64_t seq = ++tl_cross_seq_;
    {
      std::lock_guard<std::mutex> g(inbox.lock);
      inbox.entries.push_back({this->cycles() + 1, src, seq, std::move(fn)});
    }
    cross_pending_.fetch_add(1, std::memory_order_release);
  }

  struct CrossEntry {
    uint64_t due;
    uint32_t src;
    uint64_t seq;
    CrossFn  fn;
  };
  struct CrossInbox {
    std::mutex lock;
    std::vector<CrossEntry> entries;
  };
  std::vector<std::unique_ptr<CrossInbox>> cross_inbox_;
  std::atomic<uint64_t> cross_pending_{0};

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

  CrossInbox& cross_inbox_at(uint32_t d) {
    if (cross_inbox_.size() <= d) {
      cross_inbox_.resize(d + 1);
    }
    if (!cross_inbox_[d]) {
      cross_inbox_[d] = std::make_unique<CrossInbox>();
    }
    return *cross_inbox_[d];
  }

  // Run the calls due this cycle for one domain, in (due, src, seq) order —
  // the canonical order that makes serial and parallel execution agree.
  void drain_cross_calls_for(uint32_t d) {
    if (cross_inbox_.size() <= d || !cross_inbox_[d]) {
      return;
    }
    auto& inbox = *cross_inbox_[d];
    std::vector<CrossEntry> due_list;
    {
      std::lock_guard<std::mutex> g(inbox.lock);
      auto& v = inbox.entries;
      for (size_t i = 0; i < v.size();) {
        if (v[i].due <= this->cycles()) {
          due_list.push_back(std::move(v[i]));
          v[i] = std::move(v.back());
          v.pop_back();
        } else {
          ++i;
        }
      }
    }
    if (due_list.empty()) {
      return;
    }
    std::sort(due_list.begin(), due_list.end(),
              [](const CrossEntry& a, const CrossEntry& b) {
                return (a.due != b.due) ? (a.due < b.due)
                     : (a.src != b.src) ? (a.src < b.src)
                     : (a.seq < b.seq);
              });
    for (auto& e : due_list) {
      e.fn();
      cross_pending_.fetch_sub(1, std::memory_order_release);
    }
  }

  // ── Multi-threaded executor ────────────────────────────────────────────
  // One persistent worker per extra thread; domains are assigned round-robin
  // (coordinator keeps stripe 0). Two phases per cycle, separated by spinning
  // epoch-counting barriers: A) drain-inbox + tick each domain, B) deliver
  // each domain's due wheel events. All per-domain state is touched only by
  // its owner; the only shared structures are the cross inboxes (locked), the
  // inflight/cross counters (atomic), and the barrier itself.
  uint32_t mt_threads_ = 1;
  uint32_t req_workers_ = 1;
  uint32_t mt_max_threads_ = std::numeric_limits<uint32_t>::max();
  bool mt_init_done_ = false;
  bool workers_started_ = false;
  std::vector<std::thread> workers_;
  std::vector<std::vector<Domain*>> stripe_domains_;
  std::atomic<int>      barrier_count_{0};
  std::atomic<uint64_t> barrier_epoch_{0};
  std::atomic<bool>     workers_exit_{false};

  // Epoch barrier: no per-thread state, so the coordinator may be a different
  // OS thread on every call without desynchronizing the workers. The last
  // thread to arrive runs on_release before publishing the new epoch, so its
  // writes are visible to every thread on the other side of the barrier.
  template <typename Fn>
  void barrier_wait(Fn&& on_release) {
    int nthreads = int(mt_threads_);
    uint64_t epoch = barrier_epoch_.load(std::memory_order_acquire);
    if (barrier_count_.fetch_add(1, std::memory_order_acq_rel) == nthreads - 1) {
      barrier_count_.store(0, std::memory_order_relaxed);
      on_release();
      barrier_epoch_.store(epoch + 1, std::memory_order_release);
    } else {
      while (barrier_epoch_.load(std::memory_order_acquire) == epoch) {
      #if defined(__x86_64__)
        __builtin_ia32_pause();
      #endif
      }
    }
  }

  void barrier_wait() {
    this->barrier_wait([]() {});
  }

  void fire_bucket(Domain& dom, uint64_t cycle) {
    auto& bucket = dom.wheel[cycle & WHEEL_MASK];
    if (bucket.empty()) {
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
  }

  void init_mt(uint32_t num_workers) {
    mt_init_done_ = true;
    uint32_t count = std::min({std::max(1u, num_workers), num_domains_, mt_max_threads_});
    if (count < 2) {
      return;
    }
    audit_enabled_ = false;   // the audit histogram is not thread-safe
    mt_threads_ = count;
    // Materialize every domain and inbox up front: workers must never resize
    // these containers or construct a slot concurrently.
    (void)this->domain_at(num_domains_ - 1);
    for (uint32_t d = 0; d < num_domains_; ++d) {
      (void)this->cross_inbox_at(d);
    }
    stripe_domains_.assign(count, {});
    for (uint32_t d = 0; d < num_domains_; ++d) {
      stripe_domains_[d % count].push_back(domains_[d].get());
    }
    for (uint32_t w = 1; w < count; ++w) {
      workers_.emplace_back([this, w]() {
        while (true) {
          this->barrier_wait();                          // cycle start
          if (workers_exit_.load(std::memory_order_acquire)) {
            break;
          }
          // Tick and deliver back-to-back: a domain's wheel is written only
          // by its own context (cross traffic goes through the inboxes), so
          // no synchronization is needed between the two phases — each
          // worker advances its thread-local cycle view instead.
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
          this->barrier_wait([this]() { ++cycles_; });   // cycle end
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
    this->barrier_wait();       // release parked workers into the exit check
    for (auto& t : workers_) {
      t.join();
    }
    workers_.clear();
    stripe_domains_.clear();
    workers_started_ = false;
    workers_exit_.store(false, std::memory_order_relaxed);
    mt_threads_ = 1;
  }

  template <typename U> friend class SimChannel;
  template <typename U> friend class SimEventLink;
  template <typename U> friend class ::vortex::RegSlice;
  friend class SimChannelBase;
};

///////////////////////////////////////////////////////////////////////////////
// SimChannel Implementation
///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class SimChannel : public SimChannelBase {
public:
  static_assert(std::is_copy_constructible_v<Pkt>, "Packet must be copy constructible");
  using TxCallback = SmallFunction<void(const Pkt&, uint64_t), 48>;

  // Storage is unbounded; the capacity argument is accepted for API
  // compatibility and ignored.
  SimChannel(SimObjectBase* module, uint32_t /*capacity*/ = 2)
    : SimChannelBase(module)
    , pending_count_(0) {}

  SimChannel(const SimChannel& other)
    : SimChannelBase(other.module_)
    , pending_count_(0)
    , convert_fn_(other.convert_fn_)
    , tx_cb_(other.tx_cb_) {
    sink_ = nullptr;
    source_ = nullptr;
  }

  SimChannel& operator=(const SimChannel& other) {
    if (this != &other) {
      this->module_ = other.module_;
      sink_ = nullptr;
      source_ = nullptr;
      endpoint_cache_ = nullptr;
      pop_cb_ = {};
      storage_.clear();
      pending_count_ = 0;
      convert_fn_ = other.convert_fn_;
      tx_cb_ = other.tx_cb_;
    }
    return *this;
  }

  void bind(SimChannel<Pkt>* sink) {
    this->bind_setup(sink);
    convert_fn_.reset();
  }

  template <typename U, typename Converter>
  void bind(SimChannel<U>* sink, Converter&& converter) {
    static_assert(std::is_invocable_r_v<U, Converter, Pkt>, "Converter signature mismatch");
    this->bind_setup(sink);
    convert_fn_ = [sink, conv = std::forward<Converter>(converter)](const Pkt& pkt) {
      sink->receive_packet(conv(pkt));
    };
  }

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<Pkt, U>>>
  void bind(SimChannel<U>* sink) {
    this->bind_setup(sink);
    convert_fn_ = [sink](const Pkt& pkt) {
      sink->receive_packet(pkt);
    };
  }

  // Fires on delivery, in the endpoint domain's execution context: a
  // callback on a cross-domain chain must only touch endpoint-side state.
  template <typename F>
  void tx_callback(F&& callback) { tx_cb_ = std::forward<F>(callback); }

  // Back-pressure is disabled: producers never stall.
  bool full() const override { return false; }

  void send(const Pkt& pkt, uint64_t delay = 1) {
    // Unit latency: any registered delay delivers next cycle; delay 0 keeps
    // its same-cycle delta semantics.
    if (delay > 1) {
      delay = 1;
    }
    auto& platform = SimPlatform::instance();
    // A send whose endpoint lives in another domain is handed to that
    // domain's owner as a deferred call: it re-issues the send locally next
    // cycle with one less delay, landing on the same absolute cycle. All
    // channel state is thereby touched only by its owning thread.
    if (platform.is_cross_exec(this->endpoint()->module_)) {
      __assert(delay >= 1, "cross-domain send requires a registered delay");
      // Boxed: keeps the capture pointer-sized for any packet type; boundary
      // traffic is a handful of packets per cycle.
      platform.cross_call(this->endpoint()->module_->domain(),
        [this, p = std::make_shared<Pkt>(pkt), delay]() {
          this->send(*p, delay - 1);
        });
      return;
    }
    if (platform.audit_enabled()) {
      platform.audit_send(this, delay);
    }
    this->reserve();
    platform.schedule(this, pkt, delay);
  }

  void send(Pkt&& pkt, uint64_t delay = 1) {
    if (delay > 1) {
      delay = 1;
    }
    auto& platform = SimPlatform::instance();
    if (platform.is_cross_exec(this->endpoint()->module_)) {
      __assert(delay >= 1, "cross-domain send requires a registered delay");
      platform.cross_call(this->endpoint()->module_->domain(),
        [this, p = std::make_shared<Pkt>(std::move(pkt)), delay]() {
          this->send(*p, delay - 1);
        });
      return;
    }
    if (platform.audit_enabled()) {
      platform.audit_send(this, delay);
    }
    this->reserve();
    platform.schedule(this, std::move(pkt), delay);
  }

  [[nodiscard]] bool try_send(const Pkt& pkt, uint64_t delay = 1) {
    this->send(pkt, delay);
    return true;
  }

  [[nodiscard]] bool try_send(Pkt&& pkt, uint64_t delay = 1) {
    this->send(std::move(pkt), delay);
    return true;
  }

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
      return sink_->size();
    }
    assert(!SimPlatform::instance().mt_cross_access(module_));
    return this->occupancy();
  }

  // Nominal capacity, large enough to never gate and small enough that
  // credit counters derived from it (RegSlice casts to int32) stay valid.
  uint32_t capacity() const override { return (1u << 30); }

protected:
  void reserve() override {
    if (sink_) {
      sink_->reserve();
    } else {
      ++pending_count_;
      ++SimChannelBase::inflight_count();
      wake_module(module_);
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
    this->queue_push(pkt);
  }

private:
  bool forwarded() const { return sink_ != nullptr; }

  void bind_setup(SimChannelBase* sink) {
    __assert(sink != nullptr, "bind target is null");
    __assert(sink_ == nullptr, "channel already bound");
    sink_ = sink;
    sink->source_ = this;
  }

  void assert_endpoint() const {
    __assert(!forwarded(), "cannot read from a forwarded channel");
  }

  uint32_t occupancy() const { return this->queue_size() + pending_count_; }

  bool queue_empty() const { return storage_.empty(); }
  uint32_t queue_size() const { return uint32_t(storage_.size()); }
  const Pkt& queue_front() const { return storage_.front(); }
  void queue_pop() {
    storage_.pop_front();
    --SimChannelBase::inflight_count();
    if (pop_cb_) {
      pop_cb_();
    }
  }
  void queue_push(const Pkt& pkt) {
    // High-water check: with back-pressure disabled a feedback loop that
    // outruns its consumer must fail loudly instead of growing silently.
    __assert(storage_.size() < (1u << 20), "functional channel high-water exceeded");
    storage_.push_back(pkt);
  }

  std::deque<Pkt> storage_;
  uint32_t pending_count_;
  SmallFunction<void(const Pkt&), 48> convert_fn_;
  TxCallback tx_cb_;

  template <typename U> friend class SimChannel;
  template <typename U> friend class SimChannelEvent;
  friend class SimChannelBase;
};

// Constructs a std::array of N SimChannel<Pkt> instances, each bound to owner.
// SimChannel is not default-constructible, so a plain brace-init list cannot be used.
//     std::array<SimChannel<Pkt>, N> Inputs = make_sim_channels<Pkt, N>(this);

namespace detail {
template <typename Pkt, std::size_t N, std::size_t... Is>
inline std::array<SimChannel<Pkt>, N> make_sim_channels_impl(SimObjectBase* owner, std::index_sequence<Is...>) {
  return std::array<SimChannel<Pkt>, N>{ ((void)Is, SimChannel<Pkt>(owner))... };
}
} // namespace detail

template <typename Pkt, std::size_t N>
inline std::array<SimChannel<Pkt>, N> make_sim_channels(SimObjectBase* owner) {
  return detail::make_sim_channels_impl<Pkt, N>(owner, std::make_index_sequence<N>{});
}

///////////////////////////////////////////////////////////////////////////////
// SimEventLink
///////////////////////////////////////////////////////////////////////////////

// One-way, typed event link for control-plane strobes (doorbells, barrier
// arrive/resume, completion kicks). Both ends are declared members: the
// receiver binds a member-function handler to its end, and the sender's end
// is wired to the receiver's at elaboration with the same bind() idiom
// channels use (fan-in is allowed: multiple out-ends may target one handler
// end). Delivery is non-refusable and runs the bound handler in the
// receiving module's execution context, with same-cycle deliveries merged in
// canonical (due, src, seq) order — identical serial and parallel. Unlike a
// channel, a link has no queue, no occupancy, and no back-pressure; it is
// the only way to trigger behavior on a module in another execution domain.
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
  // context at the next cycle (unit latency — any declared delay collapses).
  // Cannot fail and cannot be refused.
  void send(const Msg& msg, uint64_t delay = 1) {
    __assert(delay >= 1, "event link delivery requires a registered delay");
    auto* ep = this->endpoint();
    __assert(!!ep->handler_, "event link has no bound handler");
    SimPlatform::instance().cross_call(ep->module_->domain(),
      [ep, msg]() {
        ep->handler_(msg);
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
      domain_at(obj->domain_).scan.push_back(obj.get());
    }
    if (SimObject<Impl>::template has_own_reset<Impl>()) {
      active_reset_.push_back(obj.get());
    }
  } else {
    domain_at(obj->domain_).scan.push_back(obj.get());
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
    auto evt = new SimChannelEvent<Pkt>(channel, pkt, dom.delta);
    dom.imm.push_back(evt);
    ++dom.delta;
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
    __assert(tl_exec_domain_ == HOST_DOMAIN
        || &dom == &domain_at(tl_exec_domain_), "cross-domain delta send");
    auto evt = new SimChannelEvent<Pkt>(channel, std::move(pkt), dom.delta);
    dom.imm.push_back(evt);
    ++dom.delta;
  } else {
    uint64_t fire_cycle = this->cycles() + delay;
    auto evt = new SimChannelEvent<Pkt>(channel, std::move(pkt), fire_cycle);
    dom.wheel[fire_cycle & WHEEL_MASK].push_back(evt);
  }
}

template <typename Pkt, typename Func>
void SimPlatform::schedule(Func&& func, const Pkt& pkt, uint64_t delay, uint32_t domain) {
  __assert(delay != 0, "scheduled callbacks require a registered delay");
  // Unit latency: declared callback latencies collapse like channel delays.
  auto& dom = domain_at(domain);
  uint64_t fire_cycle = this->cycles() + 1;
  auto evt = new SimCallEvent<Pkt>(std::forward<Func>(func), pkt, fire_cycle);
  dom.wheel[fire_cycle & WHEEL_MASK].push_back(evt);
}

// One-time topology validation, run at the first reset after elaboration:
// every bound channel chain crossing an execution-domain boundary must pass
// through a registered boundary stage. An unregistered crossing is safe
// serially (sends marshal through the inboxes) but couples behavior to the
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
    while (!dom->imm.empty()) {
      auto it = dom->imm.begin();
      auto evt = &*it;
      dom->imm.erase(it);
      delete evt;
    }
    dom->delta = 0;
  }

  // clear sim objects (only those that override reset())
  for (auto* object : active_reset_) {
    object->do_reset();
  }

  // Re-arm every tick-gated object for the new run.
  for (auto& object : objects_) {
    object->tick_active_ = true;
  }

  // Drop undelivered cross-domain calls from the previous run. The inbox
  // slots themselves are kept: workers index them concurrently once started,
  // so the vector must never shrink after init_mt materialized it.
  for (auto& inbox : cross_inbox_) {
    if (inbox) {
      std::lock_guard<std::mutex> g(inbox->lock);
      inbox->entries.clear();
    }
  }
  cross_pending_ = 0;

  // Reset timing
  cycles_ = 0;
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
    this->barrier_wait();                          // release workers
    uint64_t cycle = cycles_;
    tl_cycles_ = cycle;
    for (auto* dom : stripe_domains_[0]) {
      this->tick_domain(*dom);
    }
    tl_cycles_ = cycle + 1;
    for (auto* dom : stripe_domains_[0]) {
      this->fire_bucket(*dom, cycle + 1);
    }
    tl_exec_domain_ = HOST_DOMAIN;
    tl_exec_object_ = nullptr;
    this->barrier_wait([this]() { ++cycles_; });   // join; workers park
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

inline void SimPlatform::tick_domain(Domain& dom) {
  tl_exec_domain_ = dom.id;
  tl_exec_object_ = nullptr;
  // Start-of-cycle for this domain: integrate deferred cross-domain work.
  if (this->cross_pending() != 0) {
    this->drain_cross_calls_for(dom.id);
  }
  if (dom.delta != 0) {
    fire_immediate_events(dom);
  }
  for (auto* object : dom.scan) {
    if (!object->tick_active_) {
      continue;
    }
    tl_exec_domain_ = object->domain_;
    tl_exec_object_ = object;
    object->do_tick();
    if (dom.delta != 0) {
      fire_immediate_events(dom);
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
    while (!dom->imm.empty()) {
      auto it = dom->imm.begin();
      auto evt = &*it;
      dom->imm.erase(it);
      delete evt;
    }
  }
  domains_.clear();
  cross_inbox_.clear();
  cross_pending_ = 0;
  num_domains_ = 1;
  cycles_ = 0;
  // Allow a subsequent initialize() to elaborate and parallelize afresh.
  mt_init_done_ = false;
  mt_max_threads_ = std::numeric_limits<uint32_t>::max();
  topo_validated_ = false;
  audit_enabled_ = audit_requested_;
}

inline void SimPlatform::fire_immediate_events(Domain& dom) {
  // Each immediate event gets a unique, monotonically increasing delta at
  // schedule time, so the list is already ordered by delta — including events
  // appended mid-firing, which receive a delta above all queued ones. Firing
  // front-to-back is therefore identical to draining delta levels in order.
  while (!dom.imm.empty()) {
    auto it = dom.imm.begin();
    auto evt = &*it;
    evt->fire();
    dom.imm.erase(it);
    delete evt;
  }
  dom.delta = 0;
}

template <typename Pkt>
void SimChannelEvent<Pkt>::fire() {
  channel_->receive_packet(pkt_);
}

inline SimObjectBase::SimObjectBase(const SimContext&, const std::string& name)
    : name_(name)
    , domain_(SimPlatform::instance().build_domain()) {}

inline SimChannelBase::SimChannelBase(SimObjectBase* module)
    : module_(module)
    , sink_(nullptr)
    , source_(nullptr) {
  SimPlatform::instance().register_channel(this);
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

} // namespace sim_functional
} // namespace vortex
