// Copyright © 2019-2023
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

#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <list>
#include <queue>
#include <assert.h>
#include "mempool.h"
#include "util.h"
#include "linked_list.h"

namespace vortex {

class SimObjectBase;

class SimPortBase {
public:
  virtual ~SimPortBase() {}

  SimObjectBase* module() const {
    return module_;
  }

  SimPortBase* sink() const {
    return sink_;
  }

  SimPortBase* source() const {
    return source_;
  }

  virtual bool empty() const = 0;

  virtual bool full() const = 0;

  virtual uint32_t size() const = 0;

  virtual uint32_t capacity() const = 0;

protected:
  SimPortBase(SimObjectBase* module, uint32_t capacity)
    : module_(module)
    , capacity_(capacity)
    , sink_(nullptr)
    , source_(nullptr)
  {}

  virtual void do_pop() = 0;

  SimPortBase& operator=(const SimPortBase&) = delete;

  SimObjectBase* module_;
  uint32_t       capacity_;
  SimPortBase*   sink_;
  SimPortBase*   source_;

  LinkedListNode<SimPortBase> pop_list_;
  LinkedListNode<SimPortBase> push_list_;

  friend class SimPlatform;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class SimPort : public SimPortBase {
public:
  typedef std::function<void (const Pkt&, uint64_t)> TxCallback;

  SimPort(SimObjectBase* module, uint32_t capacity = 0)
    : SimPortBase(module, capacity)
    , tx_cb_(nullptr)
  {}

  void bind(SimPort<Pkt>* sink) {
    __assert(0 == capacity_, "only virtual ports can be used a link!")
    assert(sink_ == nullptr);
    sink->source_ = this;
    sink_ = sink;
    sink_transfer_ = nullptr;
  }

  template <typename U>
  void bind(SimPort<U>* sink) {
    __assert(0 == capacity_, "only virtual ports can be used a link!")
    assert(sink_ == nullptr);
    sink->source_ = this;
    sink_ = sink;
    sink_transfer_ = [sink](const Pkt& pkt, uint64_t cycles) {
      sink->transfer(static_cast<U>(pkt), cycles);
    };
  }

  template <typename U, typename Converter>
  void bind(SimPort<U>* sink, const Converter& converter) {
    __assert(0 == capacity_, "only virtual ports can be used a link!")
    assert(sink_ == nullptr);
    sink->source_ = this;
    sink_ = sink;
    sink_transfer_ = [sink, converter](const Pkt& pkt, uint64_t cycles) {
      sink->transfer(static_cast<U>(converter(pkt)), cycles);
    };
  }

  void unbind() {
    if (sink_) {
      sink_->source_ = nullptr;
      sink_ = nullptr;
      sink_transfer_ = nullptr;
    }
  }

  bool empty() const override {
    if (sink_) {
      return sink_->empty();
    }
    return queue_.empty();
  }

  bool full() const override {
    if (sink_) {
      return sink_->full();
    }
    return (capacity_ != 0 && queue_.size() >= capacity_);
  }

  uint32_t size() const override {
    if (sink_) {
      return sink_->size();
    }
    return queue_.size();
  }

  uint32_t capacity() const override {
    if (sink_) {
      return sink_->capacity();
    }
    return capacity_;
  }

  const Pkt& front() const {
    __assert(sink_ == nullptr, "cannot be called on a stub port!")
    __assert(!this->empty(), "port is empty!");
    return queue_.front();
  }

  Pkt& front() {
    __assert(sink_ == nullptr, "cannot be called on a stub port!")
    __assert(!this->empty(), "port is empty!");
    return queue_.front().pkt;
  }

  void push(const Pkt& pkt, uint64_t delay = 1);

  uint64_t pop();

  void tx_callback(const TxCallback& callback) {
    tx_cb_ = callback;
  }

protected:
  struct timed_pkt_t {
    Pkt      pkt;
    uint64_t cycles;
  };

  std::queue<timed_pkt_t> queue_;
  TxCallback tx_cb_;
  TxCallback sink_transfer_;

  void transfer(const Pkt& pkt, uint64_t cycles) {
    if (tx_cb_) {
      tx_cb_(pkt, cycles);
    }
    if (sink_) {
      if (sink_transfer_) {
        sink_transfer_(pkt, cycles);
      } else {
        reinterpret_cast<SimPort<Pkt>*>(sink_)->transfer(pkt, cycles);
      }
    } else {
      queue_.push({pkt, cycles});
    }
  }

  void do_pop() override {
    queue_.pop();
  }

  SimPort& operator=(const SimPort&) = delete;

  template <typename U> friend class SimPortEvent;
  template <typename U> friend class SimPort;
};

///////////////////////////////////////////////////////////////////////////////

class SimEventBase {
public:
  typedef std::shared_ptr<SimEventBase> Ptr;

  virtual ~SimEventBase() {}

  virtual void fire() const = 0;

  uint64_t cycles() const {
    return cycles_;
  }

protected:
  SimEventBase(uint64_t cycles) : cycles_(cycles) {}

  uint64_t cycles_;

  LinkedListNode<SimEventBase> list_;

  friend class SimPlatform;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class SimCallEvent : public SimEventBase {
public:
  void fire() const override {
    func_(pkt_);
  }

  typedef std::function<void (const Pkt&)> Func;

  SimCallEvent(const Func& func, const Pkt& pkt, uint64_t cycles)
    : SimEventBase(cycles)
    , func_(func)
    , pkt_(pkt)
  {}

  static void* operator new(std::size_t sz) {
    __unused (sz);
    assert(sizeof(SimCallEvent<Pkt>) == sz);
    return allocator_.allocate(1);
  }

  static void operator delete(void* ptr, std::size_t sz) noexcept {
    __unused (sz);
    assert(sizeof(SimCallEvent<Pkt>) == sz);
    allocator_.deallocate(static_cast<SimCallEvent<Pkt>*>(ptr), 1);
  }

protected:
  Func func_;
  Pkt  pkt_;
  static inline PoolAllocator<SimCallEvent<Pkt>, 64> allocator_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class SimPortEvent : public SimEventBase {
public:
  void fire() const override {
    const_cast<SimPort<Pkt>*>(port_)->transfer(pkt_, cycles_);
  }

  SimPortEvent(const SimPort<Pkt>* port, const Pkt& pkt, uint64_t cycles)
    : SimEventBase(cycles)
    , port_(port)
    , pkt_(pkt)
  {}

  static void* operator new(std::size_t sz) {
    __unused (sz);
    assert(sizeof(SimPortEvent<Pkt>) == sz);
    return allocator_.allocate(1);
  }

  static void operator delete(void* ptr, std::size_t sz) noexcept {
    __unused (sz);
    assert(sizeof(SimPortEvent<Pkt>) == sz);
    allocator_.deallocate(static_cast<SimPortEvent<Pkt>*>(ptr), 1);
  }

protected:
  const SimPort<Pkt>* port_;
  Pkt pkt_;
  static inline PoolAllocator<SimPortEvent<Pkt>, 64> allocator_;
};

///////////////////////////////////////////////////////////////////////////////

class SimContext {
private:
  SimContext() {}

  friend class SimPlatform;
};

///////////////////////////////////////////////////////////////////////////////

class SimObjectBase {
public:
  typedef std::shared_ptr<SimObjectBase> Ptr;

  virtual ~SimObjectBase() {}

  const std::string& name() const {
    return name_;
  }

protected:

  SimObjectBase(const SimContext&, const std::string& name) : name_(name) {}

private:

  std::string name_;

  virtual void do_reset() = 0;

  virtual void do_tick() = 0;

  friend class SimPortBase;
  friend class SimPlatform;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Impl>
class SimObject : public SimObjectBase {
public:
  typedef std::shared_ptr<Impl> Ptr;

  template <typename... Args>
  static Ptr Create(Args&&... args);

protected:

  SimObject(const SimContext& ctx, const std::string& name)
    : SimObjectBase(ctx, name)
  {}

private:

  const Impl* impl() const {
    return static_cast<const Impl*>(this);
  }

  Impl* impl() {
    return static_cast<Impl*>(this);
  }

  void do_reset() override {
    this->impl()->reset();
  }

  void do_tick() override {
    this->impl()->tick();
  }
};

///////////////////////////////////////////////////////////////////////////////

class SimPlatform {
public:
  static SimPlatform& instance() {
    static SimPlatform s_inst;
    return s_inst;
  }

  bool initialize() {
    //--
    return true;
  }

  void finalize() {
    instance().cleanup();
  }

  template <typename Impl, typename... Args>
  typename SimObject<Impl>::Ptr create_object(Args&&... args) {
    auto obj = std::make_shared<Impl>(SimContext{}, std::forward<Args>(args)...);
    objects_.push_back(obj);
    return obj;
  }

  template <typename Pkt>
  void schedule(const typename SimCallEvent<Pkt>::Func& callback,
                const Pkt& pkt,
                uint64_t delay) {
    if (delay == 0) {
      auto evt = new SimCallEvent<Pkt>(callback, pkt, delta_);
      imm_events_.push_back(evt);
      ++delta_;
    } else {
      auto evt = new SimCallEvent<Pkt>(callback, pkt, cycles_ + delay);
      reg_events_.push_back(evt);
    }
  }

  void reset() {
    assert(imm_events_.empty() && "immediate events not cleared!");
    assert(reg_events_.empty() && "registered events not cleared!");
    imm_events_.clear();
    reg_events_.clear();
    for (auto& object : objects_) {
      object->do_reset();
    }
    cycles_ = 0;
    delta_ = 0;
  }

  void tick() {
    // execute objects
    this->fire_immediate_events();
    for (auto& object : objects_) {
      object->do_tick();
      this->fire_immediate_events();
    }

    // realize objects
    for (auto it = pop_list_.begin(); it != pop_list_.end();) {
      it->do_pop();
      it = pop_list_.erase(it);
    }
    push_list_.clear();

    // fire registered events
    this->fire_registered_events();
  }

  uint64_t cycles() const {
    return cycles_;
  }

private:

  SimPlatform() : cycles_(0), delta_(0) {}

  virtual ~SimPlatform() {
    this->cleanup();
  }

  void cleanup() {
    objects_.clear();
    assert(imm_events_.empty() && "immediate events not cleared!");
    assert(reg_events_.empty() && "registered events not cleared!");
    imm_events_.clear();
    reg_events_.clear();
  }

  template <typename Pkt>
  void schedule_push(SimPort<Pkt>* port, const Pkt& pkt, uint64_t delay) {
    if (port->capacity() != 0) {
      __assert(0 == push_list_.count(port), "cannot enqueue a port multiple times during the same cycle!");
      push_list_.push_back(port);
    }
    // schedule update event
    if (delay == 0) {
      auto evt = new SimPortEvent<Pkt>(port, pkt, delta_);
      imm_events_.push_back(evt);
      ++delta_;
    } else {
      auto evt = new SimPortEvent<Pkt>(port, pkt, cycles_ + delay);
      reg_events_.push_back(evt);
    }
  }

  template <typename Pkt>
  void schedule_pop(SimPort<Pkt>* port) {
    __assert(0 == pop_list_.count(port), "cannot dequeue a port multiple times during the same cycle!");
    pop_list_.push_back(port);
  }

  void fire_immediate_events() {
    // fire all events that are scheduled for the current cycle in issue order
    for (uint32_t delta = 0; delta < delta_; ++delta) {
      for (auto evt_it = imm_events_.begin(), evt_it_end = imm_events_.end(); evt_it != evt_it_end;) {
        auto event = &*evt_it;
        if (event->cycles() == delta) {
          event->fire();
          evt_it = imm_events_.erase(evt_it);
          delete event;
        } else {
          ++evt_it;
        }
      }
    };
    delta_ = 0;
  }

  void fire_registered_events() {
    // advance the clock
    ++cycles_;

    // fire all events that are scheduled for the current cycle
    for (auto evt_it = reg_events_.begin(), evt_it_end = reg_events_.end(); evt_it != evt_it_end;) {
      auto event = &*evt_it;
      if (event->cycles() == cycles_) {
        event->fire();
        evt_it = reg_events_.erase(evt_it);
        delete event;
      } else {
        ++evt_it;
      }
    }
  }

  std::vector<SimObjectBase::Ptr> objects_;
  LinkedList<SimEventBase, &SimEventBase::list_> reg_events_;
  LinkedList<SimEventBase, &SimEventBase::list_> imm_events_;
  LinkedList<SimPortBase, &SimPortBase::push_list_> push_list_;
  LinkedList<SimPortBase, &SimPortBase::pop_list_> pop_list_;
  uint64_t cycles_;
  uint32_t delta_;

  template <typename U> friend class SimPort;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
void SimPort<Pkt>::push(const Pkt& pkt, uint64_t delay) {
  __assert(source_ == nullptr, "cannot be called on a sink port!")
  __assert(!this->full(), "port is full!");
  SimPlatform::instance().schedule_push(this, pkt, delay);
}

template <typename Pkt>
uint64_t SimPort<Pkt>::pop() {
  __assert(sink_ == nullptr, "cannot be called on a stub port!")
  __assert(!this->empty(), "port is empty!");
  SimPlatform::instance().schedule_pop(this);
  return queue_.front().cycles;
}

///////////////////////////////////////////////////////////////////////////////

template <typename Impl>
template <typename... Args>
typename SimObject<Impl>::Ptr SimObject<Impl>::Create(Args&&... args) {
  return SimPlatform::instance().create_object<Impl>(std::forward<Args>(args)...);
}

}