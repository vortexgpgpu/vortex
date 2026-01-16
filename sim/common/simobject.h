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

  // Recursive reservation for backpressure tracking
  virtual void reserve() = 0;

  SimObjectBase* module_;
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
  SimPlatform() : cycles_(0), delta_(0) {}
  ~SimPlatform() { cleanup(); }

  void cleanup();
  void fire_immediate_events();
  void fire_registered_events();

  template <typename Pkt>
  void schedule(SimChannel<Pkt>* channel, const Pkt& pkt, uint64_t delay);

  // Rvalue packet scheduling: avoids an extra copy for movable packet types.
  template <typename Pkt>
  void schedule(SimChannel<Pkt>* channel, Pkt&& pkt, uint64_t delay);

  std::vector<std::shared_ptr<SimObjectBase>> objects_;
  LinkedList<SimEventBase, &SimEventBase::list_> reg_events_;
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

  // --------------------------------------------------------------------------
  // Configuration / Binding
  // --------------------------------------------------------------------------

  // Direct bind (same packet type): fastest path (no converter stored)
  void bind(SimChannel<Pkt>* sink) {
    this->bind_setup(sink);
    convert_fn_.reset();
  }

  // Converter bind (explicit type translation)
  template <typename U, typename Converter>
  void bind(SimChannel<U>* sink, Converter&& converter) {
    static_assert(std::is_invocable_r_v<U, Converter, Pkt>, "Converter signature mismatch");
    this->bind_setup(sink);

    // Backpressure routes via sink_. Conversion happens on delivery.
    convert_fn_ = [sink, conv = std::forward<Converter>(converter)](const Pkt& pkt) {
      sink->receive_packet(conv(pkt));
    };
  }

  // Implicit conversion bind (e.g., Derived -> Base)
  template <typename U, typename = std::enable_if_t<std::is_convertible_v<Pkt, U>>>
  void bind(SimChannel<U>* sink) {
    this->bind_setup(sink);

    convert_fn_ = [sink](const Pkt& pkt) {
      sink->receive_packet(pkt); // implicit conversion
    };
  }

  template <typename F>
  void tx_callback(F&& callback) { tx_cb_ = std::forward<F>(callback); }

  // --------------------------------------------------------------------------
  // Producer API
  // --------------------------------------------------------------------------

  bool full() const override {
    // Forwarded channel: delegate full() to sink.
    if (sink_)
      return sink_->full();

    // Reservation Model counts pending packets.
    return this->occupancy() >= storage_.capacity();
  }

  void send(const Pkt& pkt, uint64_t delay = 1) {
    __assert(!this->full(), "channel is full");
    this->reserve();
    // Schedule arrival
    SimPlatform::instance().schedule(this, pkt, delay);
  }

  void send(Pkt&& pkt, uint64_t delay = 1) {
    __assert(!this->full(), "channel is full");
    this->reserve();
    // Schedule arrival
    SimPlatform::instance().schedule(this, std::move(pkt), delay);
  }

  [[nodiscard]] bool try_send(const Pkt& pkt, uint64_t delay = 1) {
    if (this->full())
      return false;
    this->send(pkt, delay);
    return true;
  }

  [[nodiscard]] bool try_send(Pkt&& pkt, uint64_t delay = 1) {
    if (this->full())
      return false;
    this->send(std::move(pkt), delay);
    return true;
  }

  // --------------------------------------------------------------------------
  // Consumer API (endpoint only)
  // --------------------------------------------------------------------------

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
    if (this->empty())
      return false;
    *out = this->peek();
    this->pop();
    return true;
  }

  uint32_t size() const override {
    if (sink_)
      return sink_->size();
    return this->occupancy();
  }

  uint32_t capacity() const override {
    if (sink_)
      return sink_->capacity();
    return storage_.capacity();
  }

protected:
  // Backpressure reservation (recursive). Endpoint increments pending_count_.
  void reserve() override {
    if (sink_) {
      sink_->reserve();
    } else {
      ++pending_count_;
    }
  }

  // Called by SimChannelEvent when the wire delay expires.
  void receive_packet(const Pkt& pkt) {
    // Optional debug callback
    if (tx_cb_) {
      tx_cb_(pkt, SimPlatform::instance().cycles());
    }
    if (sink_) {
      // Forwarded channel:
      // - If convert_fn_ is set => Convert bind
      // - Else => Direct bind (same-type)
      if (convert_fn_) {
        convert_fn_(pkt);
      } else {
        auto* sink = static_cast<SimChannel<Pkt>*>(sink_);
        sink->receive_packet(pkt);
  }
      return;
    }
    // Endpoint: move from "Wire" (pending) to "Buffer" (queue).
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

  // Queue helpers (endpoint only). Virtual channels (capacity=0) must be bound.
  bool queue_empty() const {
    return storage_.empty();
  }

  uint32_t queue_size() const {
    return storage_.size();
  }

  const Pkt& queue_front() const {
    return storage_.front();
  }

  void queue_pop() {
    storage_.pop();
  }

  void queue_push(const Pkt& pkt) {
    storage_.push(pkt);
  }
  // Uses RingQueue for fixed allocation.
  RingQueue<Pkt> storage_;

  // Track in-flight packets
  uint32_t pending_count_;

  // Optional conversion hook for forwarded channels (unset for direct binds)
  SmallFunction<void(const Pkt&), 48> convert_fn_;

  // Optional debug callback
  TxCallback tx_cb_;

  template <typename U> friend class SimChannel;
  template <typename U> friend class SimChannelEvent;
  friend class SimChannelBase;
};

///////////////////////////////////////////////////////////////////////////////
// Object Creation & Platform Implementation
///////////////////////////////////////////////////////////////////////////////

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

private:
  Impl* impl() { return static_cast<Impl*>(this); }
  void do_reset() override { impl()->reset(); }
  void do_tick()  override { impl()->tick();  }
};

template <typename Impl, typename... Args>
std::shared_ptr<Impl> SimPlatform::create_object(Args&&... args) {
  auto obj = std::make_shared<Impl>(SimContext{}, std::forward<Args>(args)...);
  objects_.push_back(obj);
  return obj;
}

template <typename Pkt>
void SimPlatform::schedule(SimChannel<Pkt>* channel, const Pkt& pkt, uint64_t delay) {
  if (delay == 0) {
    auto evt = new SimChannelEvent<Pkt>(channel, pkt, delta_);
    imm_events_.push_back(evt);
    ++delta_;
  } else {
    auto evt = new SimChannelEvent<Pkt>(channel, pkt, cycles_ + delay);
    reg_events_.push_back(evt);
  }
}
template <typename Pkt>
void SimPlatform::schedule(SimChannel<Pkt>* channel, Pkt&& pkt, uint64_t delay) {
  if (delay == 0) {
    auto evt = new SimChannelEvent<Pkt>(channel, std::move(pkt), delta_);
    imm_events_.push_back(evt);
    ++delta_;
  } else {
    auto evt = new SimChannelEvent<Pkt>(channel, std::move(pkt), cycles_ + delay);
    reg_events_.push_back(evt);
  }
}

template <typename Pkt, typename Func>
void SimPlatform::schedule(Func&& func, const Pkt& pkt, uint64_t delay) {
  if (delay == 0) {
    auto evt = new SimCallEvent<Pkt>(std::forward<Func>(func), pkt, delta_);
    imm_events_.push_back(evt);
    ++delta_;
  } else {
    auto evt = new SimCallEvent<Pkt>(std::forward<Func>(func), pkt, cycles_ + delay);
    reg_events_.push_back(evt);
  }
}

inline void SimPlatform::reset() {
  __assert(imm_events_.empty(), "reset() requires no pending events");
  __assert(reg_events_.empty(), "reset() requires no pending events");
  cycles_ = 0;
  delta_ = 0;
  for (auto& object : objects_) object->do_reset();
}

inline void SimPlatform::tick() {
  fire_immediate_events();
  for (auto& object : objects_) {
    object->do_tick();
    fire_immediate_events();
  }
  ++cycles_;
  fire_registered_events();
}

inline void SimPlatform::cleanup() {
  objects_.clear();
  while(!reg_events_.empty()) {
    auto it = reg_events_.begin();
    auto evt = &*it;
    reg_events_.erase(it);
    delete evt;
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

inline void SimPlatform::fire_registered_events() {
  for (auto it = reg_events_.begin(); it != reg_events_.end();) {
    auto evt = &*it;
    if (evt->cycles() <= cycles_) {
      evt->fire();
      it = reg_events_.erase(it);
      delete evt;
    } else {
      ++it;
    }
  }
}

template <typename Pkt>
void SimChannelEvent<Pkt>::fire() {
  channel_->receive_packet(pkt_);
}

} // namespace vortex
