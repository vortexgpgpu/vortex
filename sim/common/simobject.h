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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "linked_list.h"
#include "mempool.h"
#include "util.h"
#include "smallfunc.h"
#include "ringqueue.h"

namespace vortex {

class SimContext {
private:
  SimContext() = default;
  friend class SimPlatform;
};

// Base class for all simulation objects (Nodes)
class SimObjectBase {
public:
  using Ptr = std::shared_ptr<SimObjectBase>;
  virtual ~SimObjectBase() = default;

  const std::string& name() const { return name_; }

protected:
  SimObjectBase(const SimContext&, const std::string& name) : name_(name) {}

private:
  std::string name_;
  virtual void do_reset() = 0;
  virtual void do_tick()  = 0;
  friend class SimPlatform;
};

// Base class for channels (Topological introspection)
class SimChannelBase {
public:
  virtual ~SimChannelBase() = default;

  // Introspection API
  SimObjectBase* module() const { return module_; }
  SimChannelBase* sink() const { return sink_; }
  SimChannelBase* source() const { return source_; }

  virtual bool empty() const = 0;
  virtual bool full() const = 0;
  virtual uint32_t size() const = 0;
  virtual uint32_t capacity() const = 0;

protected:
  explicit SimChannelBase(SimObjectBase* module)
      : module_(module)
      , sink_(nullptr)
      , source_(nullptr)
  {}

  virtual void reserve() = 0;

  SimObjectBase*  module_;
  SimChannelBase* sink_;
  SimChannelBase* source_;

  friend class SimPlatform;
  template <typename T> friend class SimChannel;
};

///////////////////////////////////////////////////////////////////////////////
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
  Pkt  pkt_;
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

  uint64_t cycles() const { return cycles_; }

  // Factory
  template <typename Impl, typename... Args>
  std::shared_ptr<Impl> create_object(Args&&... args);

  // Scheduling API
  template <typename Pkt, typename Func>
  void schedule(Func&& func, const Pkt& pkt, uint64_t delay);

  void reset();
  void tick();

private:
  // Timing Wheel Configuration
  static constexpr uint64_t WHEEL_SIZE = 4096;
  static constexpr uint64_t WHEEL_MASK = WHEEL_SIZE - 1;

  SimPlatform() : reg_events_(WHEEL_SIZE), cycles_(0), delta_(0) {}
  ~SimPlatform() { cleanup(); }

  void cleanup();
  void fire_immediate_events();

  template <typename Pkt>
  void schedule(SimChannel<Pkt>* channel, const Pkt& pkt, uint64_t delay);

  template <typename Pkt>
  void schedule(SimChannel<Pkt>* channel, Pkt&& pkt, uint64_t delay);

  std::vector<std::shared_ptr<SimObjectBase>> objects_;
  // Subset views over `objects_` for the per-cycle hot path. A SimObject is
  // added here only if it overrides on_tick()/on_reset() (auto-detected at
  // create_object<Impl>() time via member-pointer comparison). Passive
  // SimObjects pay no per-cycle cost.
  std::vector<SimObjectBase*> active_tick_;
  std::vector<SimObjectBase*> active_reset_;

  std::vector<LinkedList<SimEventBase, &SimEventBase::list_>> reg_events_;

  LinkedList<SimEventBase, &SimEventBase::list_> imm_events_;

  uint64_t cycles_;
  uint32_t delta_;

  template <typename U> friend class SimChannel;
};

///////////////////////////////////////////////////////////////////////////////
// SimChannel Implementation
///////////////////////////////////////////////////////////////////////////////

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
      storage_ = RingQueue<Pkt>(other.storage_.capacity());
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

  template <typename F>
  void tx_callback(F&& callback) { tx_cb_ = std::forward<F>(callback); }

  bool full() const override {
    if (sink_) return sink_->full();
    return this->occupancy() >= storage_.capacity();
  }

  void send(const Pkt& pkt, uint64_t delay = 1) {
    __assert(!this->full(), "channel is full");
    this->reserve();
    SimPlatform::instance().schedule(this, pkt, delay);
  }

  void send(Pkt&& pkt, uint64_t delay = 1) {
    __assert(!this->full(), "channel is full");
    this->reserve();
    SimPlatform::instance().schedule(this, std::move(pkt), delay);
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
    if (sink_) return sink_->size();
    return this->occupancy();
  }

  uint32_t capacity() const override {
    if (sink_) return sink_->capacity();
    return storage_.capacity();
  }

protected:
  void reserve() override {
    if (sink_) {
      sink_->reserve();
    } else {
      ++pending_count_;
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
  uint32_t queue_size() const { return storage_.size(); }
  const Pkt& queue_front() const { return storage_.front(); }
  void queue_pop() { storage_.pop(); }
  void queue_push(const Pkt& pkt) { storage_.push(pkt); }

  RingQueue<Pkt> storage_;
  uint32_t pending_count_;
  SmallFunction<void(const Pkt&), 48> convert_fn_;
  TxCallback tx_cb_;

  template <typename U> friend class SimChannel;
  template <typename U> friend class SimChannelEvent;
  friend class SimChannelBase;
};

///////////////////////////////////////////////////////////////////////////////
// Object Creation & Platform Implementation
///////////////////////////////////////////////////////////////////////////////

// Compile-time check that `on_tick()` / `on_reset()` are not publicly
// callable on `T`. Function-template SFINAE: the (int) overload is selected
// only when the call expression is well-formed at this (namespace-scope,
// non-friend) context — i.e. when the member is public. Otherwise the
// (...) fallback wins. Used by create_object<Impl> to reject derivatives
// that leave the lifecycle hooks public.
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

  // SimObject lifecycle callbacks. Protected so that only SimPlatform (via
  // do_tick/do_reset below) and the derived class itself can invoke them.
  // External code that writes `obj->on_tick()` is rejected at access-check
  // time, and create_object<Impl>() also static_asserts they aren't public.
  // Each derivative must `friend class SimObject<Self>` so that the auto-
  // detection below can compare member pointers across access boundaries.
  void on_tick()  {}
  void on_reset() {}

private:
  Impl* impl() { return static_cast<Impl*>(this); }
  void do_reset() override { impl()->on_reset(); }
  void do_tick()  override { impl()->on_tick();  }

  // Auto-detect whether `Impl` overrides on_tick()/on_reset() vs inheriting
  // the SimObject<Impl> defaults. When inherited, &Impl::on_tick resolves to
  // the same member as &SimObject<Impl>::on_tick and the cast-then-compare
  // yields equal. When overridden, the values differ.
  //
  // Defined as a static member of SimObject<Impl> so the &Impl::on_tick
  // and &SimObject<Impl>::on_tick references can see the protected members
  // (SimObject<Impl> is the friend granter / declarer).
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

// Detection trait for "is this an immediate SimObject<Impl> CRTP derivative?"
// Multi-level CRTP (e.g. Derived → Intermediate → SimObject<Intermediate>)
// makes SimObject<Derived> not a base of Derived, so the static_cast probe
// fails for those — we treat them conservatively as active.
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
  // Auto-skip optimisation only applies to direct SimObject<Impl> CRTP
  // derivatives; multi-level CRTP derivatives are conservatively kept
  // active. `if constexpr` avoids instantiating has_own_*<Impl> for the
  // multi-level case, where &Impl::on_tick is inaccessible to the trait.
  if constexpr (has_direct_simobject_base<Impl>::value) {
    if (SimObject<Impl>::template has_own_tick<Impl>())
      active_tick_.push_back(obj.get());
    if (SimObject<Impl>::template has_own_reset<Impl>())
      active_reset_.push_back(obj.get());
  } else {
    active_tick_.push_back(obj.get());
    active_reset_.push_back(obj.get());
  }
  return obj;
}

template <typename Pkt>
void SimPlatform::schedule(SimChannel<Pkt>* channel, const Pkt& pkt, uint64_t delay) {
  if (delay == 0) {
    auto evt = new SimChannelEvent<Pkt>(channel, pkt, delta_);
    imm_events_.push_back(evt);
    ++delta_;
  } else {
    uint64_t fire_cycle = cycles_ + delay;
    auto evt = new SimChannelEvent<Pkt>(channel, pkt, fire_cycle);
    reg_events_[fire_cycle & WHEEL_MASK].push_back(evt);
  }
}

template <typename Pkt>
void SimPlatform::schedule(SimChannel<Pkt>* channel, Pkt&& pkt, uint64_t delay) {
  if (delay == 0) {
    auto evt = new SimChannelEvent<Pkt>(channel, std::move(pkt), delta_);
    imm_events_.push_back(evt);
    ++delta_;
  } else {
    uint64_t fire_cycle = cycles_ + delay;
    auto evt = new SimChannelEvent<Pkt>(channel, std::move(pkt), fire_cycle);
    reg_events_[fire_cycle & WHEEL_MASK].push_back(evt);
  }
}

template <typename Pkt, typename Func>
void SimPlatform::schedule(Func&& func, const Pkt& pkt, uint64_t delay) {
  if (delay == 0) {
    auto evt = new SimCallEvent<Pkt>(std::forward<Func>(func), pkt, delta_);
    imm_events_.push_back(evt);
    ++delta_;
  } else {
    uint64_t fire_cycle = cycles_ + delay;
    auto evt = new SimCallEvent<Pkt>(std::forward<Func>(func), pkt, fire_cycle);
    reg_events_[fire_cycle & WHEEL_MASK].push_back(evt);
  }
}

inline void SimPlatform::reset() {
  // Clear any lingering events from the previous run
  for (auto& bucket : reg_events_) {
    while(!bucket.empty()) {
      auto it = bucket.begin();
      auto evt = &*it;
      bucket.erase(it);
      delete evt; // Return to pool
    }
  }

  // Clear immediate events
  while(!imm_events_.empty()) {
    auto it = imm_events_.begin();
    auto evt = &*it;
    imm_events_.erase(it);
    delete evt;
  }

  // clear sim objects (only those that override reset())
  for (auto* object : active_reset_) {
    object->do_reset();
  }

  // Reset timing
  cycles_ = 0;
  delta_ = 0;
}

inline void SimPlatform::tick() {
  // Process immediate events first
  fire_immediate_events();
  // Tick only objects that override tick() (auto-detected at create_object).
  for (auto* object : active_tick_) {
    object->do_tick();
    fire_immediate_events();
  }
  ++cycles_;

  // Process registered events
  auto& bucket = reg_events_[cycles_ & WHEEL_MASK];
  if (!bucket.empty()) {
    for (auto it = bucket.begin(); it != bucket.end();) {
      auto evt = &*it;
      if (evt->cycles() <= cycles_) {
        evt->fire();
        it = bucket.erase(it);
        delete evt; // Returns to PoolAllocator
      } else {
        ++it; // Future wraparound event
      }
    }
  }
}

inline void SimPlatform::cleanup() {
  active_tick_.clear();
  active_reset_.clear();
  objects_.clear();

  for (auto& bucket : reg_events_) {
    while(!bucket.empty()) {
      auto it = bucket.begin();
      auto evt = &*it;
      bucket.erase(it);
      delete evt;
    }
  }

  while(!imm_events_.empty()) {
    auto it = imm_events_.begin();
    auto evt = &*it;
    imm_events_.erase(it);
    delete evt;
  }
  cycles_ = 0;
  delta_ = 0;
}

inline void SimPlatform::fire_immediate_events() {
  for (uint32_t d = 0; d < delta_; ++d) {
    for (auto it = imm_events_.begin(); it != imm_events_.end();) {
      auto evt = &*it;
      if (evt->cycles() == d) {
        evt->fire();
        it = imm_events_.erase(it);
        delete evt;
      } else {
        ++it;
      }
    }
  }
  delta_ = 0;
}

template <typename Pkt>
void SimChannelEvent<Pkt>::fire() {
  channel_->receive_packet(pkt_);
}

} // namespace vortex