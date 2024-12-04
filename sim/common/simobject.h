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

class SimObjectBase;

///////////////////////////////////////////////////////////////////////////////

class SimPortBase {
public:
  virtual ~SimPortBase() {}

  SimObjectBase* module() const {
    return module_;
  }

protected:
  SimPortBase(SimObjectBase* module)
    : module_(module)
  {}

  SimPortBase& operator=(const SimPortBase&) = delete;

  SimObjectBase* module_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class SimPort : public SimPortBase {
public:
  typedef std::function<void (const Pkt&, uint64_t)> TxCallback;

  SimPort(SimObjectBase* module)
    : SimPortBase(module)
    , peer_(nullptr)
    , tx_cb_(nullptr)
  {}

  void bind(SimPort<Pkt>* peer) {
    assert(peer_ == nullptr);
    peer_ = peer;
  }

  void unbind() {
    peer_ = nullptr;
  }

  bool connected() const {
    return (peer_ != nullptr);
  }

  SimPort* peer() const {
    return peer_;
  }

  bool empty() const {
    return queue_.empty();
  }

  const Pkt& front() const {
    return queue_.front();
  }

  Pkt& front() {
    return queue_.front().pkt;
  }

  void push(const Pkt& pkt, uint64_t delay = 1) const;

  uint64_t pop() {
    auto cycles = queue_.front().cycles;
    queue_.pop();
    return cycles;
  }

  void tx_callback(const TxCallback& callback) {
    tx_cb_ = callback;
  }

  uint64_t arrival_time() const {
    if (queue_.empty())
      return 0;
    return queue_.front().cycles;
  }

protected:
  struct timed_pkt_t {
    Pkt      pkt;
    uint64_t cycles;
  };

  std::queue<timed_pkt_t> queue_;
  SimPort*   peer_;
  TxCallback tx_cb_;

  void transfer(const Pkt& data, uint64_t cycles) {
    if (tx_cb_) {
      tx_cb_(data, cycles);
    }
    if (peer_) {
      peer_->transfer(data, cycles);
    } else {
      queue_.push({data, cycles});
    }
  }

  SimPort& operator=(const SimPort&) = delete;

  template <typename U> friend class SimPortEvent;
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

  void* operator new(size_t /*size*/) {
    return allocator_.allocate();
  }

  void operator delete(void* ptr) {
    allocator_.deallocate(ptr);
  }

protected:
  Func func_;
  Pkt  pkt_;

  static MemoryPool<SimCallEvent<Pkt>> allocator_;
};

template <typename Pkt>
MemoryPool<SimCallEvent<Pkt>> SimCallEvent<Pkt>::allocator_(64);

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

  void* operator new(size_t /*size*/) {
    return allocator_.allocate();
  }

  void operator delete(void* ptr) {
    allocator_.deallocate(ptr);
  }

protected:
  const SimPort<Pkt>* port_;
  Pkt pkt_;

  static MemoryPool<SimPortEvent<Pkt>> allocator_;
};

template <typename Pkt>
MemoryPool<SimPortEvent<Pkt>> SimPortEvent<Pkt>::allocator_(64);

///////////////////////////////////////////////////////////////////////////////

class SimContext;

class SimObjectBase {
public:
  typedef std::shared_ptr<SimObjectBase> Ptr;

  virtual ~SimObjectBase() {}

  const std::string& name() const {
    return name_;
  }

protected:

  SimObjectBase(const SimContext& ctx, const std::string& name);

private:

  virtual void do_reset() = 0;

  virtual void do_tick() = 0;

  std::string name_;

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

class SimContext {
private:
  SimContext() {}

  friend class SimPlatform;
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
    instance().clear();
  }

  template <typename Impl, typename... Args>
  typename SimObject<Impl>::Ptr create_object(Args&&... args) {
    auto obj = std::make_shared<Impl>(SimContext{}, std::forward<Args>(args)...);
    objects_.push_back(obj);
    return obj;
  }

  void release_object(const SimObjectBase::Ptr& object) {
    objects_.remove(object);
  }

  template <typename Pkt>
  void schedule(const typename SimCallEvent<Pkt>::Func& callback,
                const Pkt& pkt,
                uint64_t delay) {
    assert(delay != 0);
    auto evt = std::make_shared<SimCallEvent<Pkt>>(callback, pkt, cycles_ + delay);
    events_.emplace_back(evt);
  }

  void reset() {
    events_.clear();
    for (auto& object : objects_) {
      object->do_reset();
    }
    cycles_ = 0;
  }

  void tick() {
    // evaluate events
    auto evt_it = events_.begin();
    auto evt_it_end = events_.end();
    while (evt_it != evt_it_end) {
      auto& event = *evt_it;
      if (cycles_ >= event->cycles()) {
        event->fire();
        evt_it = events_.erase(evt_it);
      } else {
        ++evt_it;
      }
    }
    // evaluate components
    for (auto& object : objects_) {
      object->do_tick();
    }
    // advance clock
    ++cycles_;
  }

  uint64_t cycles() const {
    return cycles_;
  }

private:

  SimPlatform() : cycles_(0) {}

  virtual ~SimPlatform() {
    this->clear();
  }

  void clear() {
    objects_.clear();
    events_.clear();
  }

  template <typename Pkt>
  void schedule(const SimPort<Pkt>* port, const Pkt& pkt, uint64_t delay) {
    assert(delay != 0);
    auto evt = SimEventBase::Ptr(new SimPortEvent<Pkt>(port, pkt, cycles_ + delay));
    events_.emplace_back(evt);
  }

  std::list<SimObjectBase::Ptr> objects_;
  std::list<SimEventBase::Ptr> events_;
  uint64_t cycles_;

  template <typename U> friend class SimPort;
  friend class SimObjectBase;
};

///////////////////////////////////////////////////////////////////////////////

inline SimObjectBase::SimObjectBase(const SimContext&, const std::string& name)
  : name_(name)
{}

template <typename Impl>
template <typename... Args>
typename SimObject<Impl>::Ptr SimObject<Impl>::Create(Args&&... args) {
  return SimPlatform::instance().create_object<Impl>(std::forward<Args>(args)...);
}

template <typename Pkt>
void SimPort<Pkt>::push(const Pkt& pkt, uint64_t delay) const {
  if (peer_ && !tx_cb_) {
    reinterpret_cast<const SimPort<Pkt>*>(peer_)->push(pkt, delay);
  } else {
    SimPlatform::instance().schedule(this, pkt, delay);
  }
}
