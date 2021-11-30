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

  void send(const Pkt& pkt, uint64_t delay = 1) const;

  void bind(SimPort<Pkt>* peer) {
    assert(peer_ == nullptr);
    peer_ = peer;
  }

  void unbind() {    
    assert(peer_ == nullptr);
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

  const Pkt& back() const {
    return queue_.back();
  }

  Pkt& back() {
    return queue_.back().pkt;
  }

  uint64_t pop() {
    auto cycle = queue_.front().cycle;
    queue_.pop();
    return cycle;
  }  

  void tx_callback(const TxCallback& callback) {
    tx_cb_ = callback;
  }

protected:
  struct timed_pkt_t {
    Pkt      pkt;
    uint64_t cycle;
  };

  std::queue<timed_pkt_t> queue_;
  SimPort*   peer_;
  TxCallback tx_cb_;

  void push(const Pkt& data, uint64_t cycle) {
    if (tx_cb_) {
      tx_cb_(data, cycle);
    }
    if (peer_) {
      peer_->push(data, cycle);
    } else {
      queue_.push({data, cycle});
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
  
  virtual void fire() const  = 0;

  uint64_t time() const {
    return time_;
  }

protected:
  SimEventBase(uint64_t time) : time_(time) {}

  uint64_t time_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class SimCallEvent : public SimEventBase {
public:
  void fire() const override {
    func_(pkt_);
  }

  typedef std::function<void (const Pkt&)> Func;

  SimCallEvent(const Func& func, const Pkt& pkt, uint64_t time) 
    : SimEventBase(time)
    , func_(func)
    , pkt_(pkt)
  {}

  void* operator new(size_t /*size*/) {
    return allocator().allocate();
  }

  void operator delete(void* ptr) {
    allocator().deallocate(ptr);
  }

protected:
  Func func_;
  Pkt  pkt_;

  static MemoryPool<SimCallEvent<Pkt>>& allocator() {
    static MemoryPool<SimCallEvent<Pkt>> instance(64);
    return instance;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class SimPortEvent : public SimEventBase {
public:
  void fire() const override {
    const_cast<SimPort<Pkt>*>(port_)->push(pkt_, time_);
  }

  SimPortEvent(const SimPort<Pkt>* port, const Pkt& pkt, uint64_t time) 
    : SimEventBase(time) 
    , port_(port)
    , pkt_(pkt)
  {}

  void* operator new(size_t /*size*/) {
    return allocator().allocate();
  }

  void operator delete(void* ptr) {
    allocator().deallocate(ptr);
  }

protected:
  const SimPort<Pkt>* port_; 
  Pkt pkt_;

  static MemoryPool<SimPortEvent<Pkt>>& allocator() {
    static MemoryPool<SimPortEvent<Pkt>> instance(64);
    return instance;
  }
};

///////////////////////////////////////////////////////////////////////////////

class SimContext;

class SimObjectBase {
public:
  typedef std::shared_ptr<SimObjectBase> Ptr;

  virtual ~SimObjectBase() {}

  const std::string& name() const {
    return name_;
  }

  virtual void step(uint64_t cycle) = 0;

protected:

  SimObjectBase(const SimContext& ctx, const char* name); 

  std::string name_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Impl>
class SimObject : public SimObjectBase {
public:
  typedef std::shared_ptr<Impl> Ptr;

  template <typename... Args>
  static Ptr Create(Args&&... args);

protected:

  SimObject(const SimContext& ctx, const char* name) 
    : SimObjectBase(ctx, name) 
  {}

  void step(uint64_t cycle) override {
    this->impl().step(cycle);
  }

private:

  const Impl& impl() const {
    return static_cast<const Impl&>(*this);
  }

  Impl& impl() {
    return static_cast<Impl&>(*this);
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

  void flush() {
    instance().clear();
  }

  void finalize() {
    instance().clear();
  }

  template <typename Impl, typename... Args>
  typename SimObject<Impl>::Ptr CreateObject(Args&&... args) {
    auto obj = std::make_shared<Impl>(SimContext{}, std::forward<Args>(args)...);
    objects_.push_back(obj);
    return obj;
  }

  template <typename Pkt>
  void schedule(const typename SimCallEvent<Pkt>::Func& callback,
                const Pkt& pkt, 
                uint64_t delay) {    
    assert(delay != 0);
    auto evt = std::make_shared<SimCallEvent<Pkt>>(callback, pkt, cycles_ + delay);    
    events_.emplace_back(evt);
  }

  void step() {
    // evaluate events
    auto evt_it = events_.begin();
    auto evt_it_end = events_.end();
    while (evt_it != evt_it_end) {
      auto& event = *evt_it;
      if (cycles_ >= event->time()) {        
        event->fire();
        evt_it = events_.erase(evt_it);
      } else {        
        ++evt_it;
      }
    }
    // evaluate components
    for (auto& object : objects_) {
      object->step(cycles_);
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

  std::vector<SimObjectBase::Ptr> objects_;
  std::list<SimEventBase::Ptr> events_;
  uint64_t cycles_;

  template <typename U> friend class SimPort;
  friend class SimObjectBase;
};

///////////////////////////////////////////////////////////////////////////////

inline SimObjectBase::SimObjectBase(const SimContext&, const char* name) 
  : name_(name) 
{}

template <typename Impl>
template <typename... Args>
typename SimObject<Impl>::Ptr SimObject<Impl>::Create(Args&&... args) {
  return SimPlatform::instance().CreateObject<Impl>(std::forward<Args>(args)...);
}

template <typename Pkt>
void SimPort<Pkt>::send(const Pkt& pkt, uint64_t delay) const {
  if (peer_ && !tx_cb_) {
    reinterpret_cast<const SimPort<Pkt>*>(peer_)->send(pkt, delay);    
  } else {
    SimPlatform::instance().schedule(this, pkt, delay);
  }  
}