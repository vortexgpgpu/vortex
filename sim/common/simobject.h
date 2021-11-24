#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <list>
#include <queue>
#include <assert.h>

class SimObjectBase;

///////////////////////////////////////////////////////////////////////////////

class SimPortBase {
public:  
  virtual ~SimPortBase() {}
  
  SimObjectBase* module() const {
    return module_;
  }

  SimPortBase* peer() const {
    return peer_;
  }

  bool connected() const {
    return (peer_ != nullptr);
  }

protected:
  SimPortBase(SimObjectBase* module)
    : module_(module)
    , peer_(nullptr)
  {}

  void connect(SimPortBase* peer) {
    assert(peer_ == nullptr);
    peer_ = peer;
  }

  void disconnect() {    
    assert(peer_ == nullptr);  
    peer_ = nullptr;
  }

  SimPortBase& operator=(const SimPortBase&) = delete;

  SimObjectBase* module_;
  SimPortBase*   peer_;

  template <typename U> friend class SlavePort;
  template <typename U> friend class MasterPort;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class SimPort : public SimPortBase {
public:
  void send(const Pkt& pkt, uint64_t delay) const;

  void bind(SimPort<Pkt>* peer) {
    this->connect(peer);
  }

  void unbind() {    
    this->disconnect();
  }

  bool empty() const {
    return queue_.empty();
  }

  const Pkt& top() const {
    return queue_.front();
  }

  Pkt& top() {
    return queue_.front();
  }

  void pop() {
    queue_.pop();
  } 

protected:
  SimPort(SimObjectBase* module)
    : SimPortBase(module)
  {}

  void push(const Pkt& data) {
    queue_.push(data);
  }

  SimPort& operator=(const SimPort&) = delete;

  std::queue<Pkt> queue_;

  template <typename U> friend class SimPortEvent;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class SlavePort : public SimPort<Pkt> {
public:
  SlavePort(SimObjectBase* module) : SimPort<Pkt>(module) {}  

protected:
  SlavePort& operator=(const SlavePort&) = delete;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class MasterPort : public SimPort<Pkt> {
public:
  MasterPort(SimObjectBase* module) : SimPort<Pkt>(module) {}

protected:
  MasterPort& operator=(const MasterPort&) = delete;
};

///////////////////////////////////////////////////////////////////////////////

class SimEventBase {
public:
  typedef std::shared_ptr<SimEventBase> Ptr;

  virtual ~SimEventBase() {}
  
  virtual void fire() const  = 0;

  bool step() {
    return (0 == --delay_);
  }

protected:
  SimEventBase(uint64_t delay) : delay_(delay) {}

  uint64_t delay_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class SimCallEvent : public SimEventBase {
public:
  typedef std::function<void (const Pkt&)> Func;

  template <typename... Args>
  static Ptr Create(const Func& func, const Pkt& pkt, uint64_t delay) {
    return std::make_shared<SimCallEvent>(func, pkt, delay);
  }   

  SimCallEvent(const Func& func, const Pkt& pkt, uint64_t delay) 
    : SimEventBase(delay)
    , func_(func)
    , pkt_(pkt)
  {}

  void fire() const override {
    func_(pkt_);
  }

protected:  
  Func func_;
  Pkt  pkt_; 
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class SimPortEvent : public SimEventBase {
public:
  static Ptr Create(const SimPort<Pkt>* port, const Pkt& pkt, uint64_t delay) {
    return std::make_shared<SimPortEvent>(port, pkt, delay);
  }

  SimPortEvent(const SimPort<Pkt>* port, const Pkt& pkt, uint64_t delay) 
    : SimEventBase(delay) 
    , port_(port)
    , pkt_(pkt)
  {}
  
  void fire() const override {
    const_cast<SimPort<Pkt>*>(port_)->push(pkt_);
  }

private:  
  const SimPort<Pkt>* port_; 
  Pkt pkt_;
};

///////////////////////////////////////////////////////////////////////////////

class SimContext;

class SimObjectBase {
public:
  typedef std::shared_ptr<SimObjectBase> Ptr;

  virtual ~SimObjectBase() {}

  template <typename T, typename Pkt>
  void schedule(T *obj, void (T::*entry)(const Pkt&), const Pkt& pkt, uint64_t delay);

  const std::string& name() const {
    return name_;
  }

protected:

  virtual void step(uint64_t cycle) = 0;

  SimObjectBase(const SimContext& ctx, const char* name);

private:
  std::string name_;

  friend class SimPlatform;
  friend class SimPortBase;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Impl>
class SimObject : public SimObjectBase {
public:
  typedef std::shared_ptr<Impl> Ptr;  

  template <typename... Args>
  static Ptr Create(Args&&... args);

protected:

  SimObject(const SimContext& ctx, const char* name) : SimObjectBase(ctx, name) {}

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
  template <typename Impl> template <typename... Args> 
  friend typename SimObject<Impl>::Ptr SimObject<Impl>::Create(Args&&... args);
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

  void register_object(const SimObjectBase::Ptr& obj) {
    objects_.push_back(obj);
  }

  template <typename Pkt>
  void schedule(const typename SimCallEvent<Pkt>::Func& callback, 
                const Pkt& pkt, 
                uint64_t delay) {    
    auto evt = SimCallEvent<Pkt>::Create(callback, pkt, delay);
    assert(delay != 0);
    events_.emplace_back(evt);
  }

  template <typename Pkt>
  void schedule(const SimPort<Pkt>* port, 
                const Pkt& pkt, 
                uint64_t delay) {
    auto evt = SimPortEvent<Pkt>::Create(port, pkt, delay);
    assert(delay != 0);
    events_.emplace_back(evt);
  }

  void step() {
    // evaluate events
    auto evt_it = events_.begin();
    auto evt_it_end = events_.end();
    while (evt_it != evt_it_end) {
      auto& event = *evt_it;
      if (event->step()) {        
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

  std::vector<SimObjectBase::Ptr> objects_;
  std::list<SimEventBase::Ptr> events_;
  uint64_t cycles_;
};

///////////////////////////////////////////////////////////////////////////////

inline SimObjectBase::SimObjectBase(const SimContext&, const char* name) 
  : name_(name) 
{}

template <typename Impl>
template <typename... Args>
typename SimObject<Impl>::Ptr SimObject<Impl>::Create(Args&&... args) {
  auto obj = std::make_shared<Impl>(SimContext{}, std::forward<Args>(args)...);
  SimPlatform::instance().register_object(obj);
  return obj;
}

template <typename Pkt>
void SimPort<Pkt>::send(const Pkt& pkt, uint64_t delay) const {
  if (peer_) {
    reinterpret_cast<const SimPort<Pkt>*>(peer_)->send(pkt, delay);    
  } else {
    SimPlatform::instance().schedule(this, pkt, delay);
  }  
}

template <typename T, typename Pkt>
void SimObjectBase::schedule(T *obj, void (T::*entry)(const Pkt&), const Pkt& pkt, uint64_t delay) {
  auto callback = std::bind(entry, obj, std::placeholders::_1);
  SimPlatform::instance().schedule(callback, pkt, delay);
}