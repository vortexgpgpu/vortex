#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <list>
#include <assert.h>

namespace vortex {

class SimObjectBase;

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
class SimSimpleEvent : public SimEventBase {
public:
  typedef std::function<void (const Pkt&)> Func;

  template <typename... Args>
  static Ptr Create(const Func& func, const Pkt& pkt, uint64_t delay) {
    return std::make_shared<SimSimpleEvent>(func, pkt, delay);
  }   

  SimSimpleEvent(const Func& func, const Pkt& pkt, uint64_t delay) 
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
  typedef std::function<void (const Pkt&, uint32_t)> Func;

  template <typename... Args>
  static Ptr Create(const Func& func, const Pkt& pkt, uint32_t port_id, uint64_t delay) {
    return std::make_shared<SimPortEvent>(func, pkt, port_id, delay);
  }

  SimPortEvent(const Func& func, const Pkt& pkt, uint32_t port_id, uint64_t delay) 
    : SimEventBase(delay) 
    , func_(func)
    , pkt_(pkt)
    , port_id_(port_id)
  {}
  
  void fire() const override {
    func_(pkt_, port_id_);
  }

private:  
  Func     func_;
  Pkt      pkt_;  
  uint32_t port_id_;
};

///////////////////////////////////////////////////////////////////////////////

class SimPortBase {
public:
  typedef std::shared_ptr<SimPortBase> Ptr;  

  virtual ~SimPortBase() {}
  
  SimObjectBase* module() const {
    return module_;
  }
  
  uint32_t port_id() const {
    return port_id_;
  }

  SimPortBase* peer() const {
    return peer_;
  }

  bool connected() const {
    return (peer_ != nullptr);
  }

  bool is_slave() const {
    return is_slave_;
  }

protected:

  SimPortBase(SimObjectBase* module, bool is_slave);

  void connect(SimPortBase* peer) {
    assert(peer_ == nullptr);
    peer_ = peer;
  }

  void disconnect() { 
    assert(peer_ == nullptr);  
    peer_ = nullptr;
  }

  SimObjectBase* module_;
  uint32_t       port_id_;
  bool           is_slave_;
  SimPortBase*   peer_;

  template <typename Pkt> friend class MasterPort;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class SlavePort : public SimPortBase {
public:
  typedef std::shared_ptr<SlavePort<Ptr>> Ptr;
  typedef std::function<void (const Pkt&, uint32_t)> Func;

  static Ptr Create(SimObjectBase* module, const Func& func) {
    return std::make_shared<SlavePort<Pkt>>(module, func);
  }

  template <typename T>
  static Ptr Create(SimObjectBase* module, T *obj, void (T::*entry)(const Pkt&, uint32_t)) {
    return std::make_shared<SlavePort<Pkt>>(module, obj, entry);
  } 

  SlavePort(SimObjectBase* module, const Func& func)
    : SimPortBase(module, true)
    , func_(func)
  {}

  template <typename T>
  SlavePort(SimObjectBase* module, T *obj, void (T::*entry)(const Pkt&, uint32_t))
    : SimPortBase(module, true)
    , func_(std::bind(entry, obj, std::placeholders::_1, std::placeholders::_2))
  {}

  SlavePort(SimObjectBase* module, SlavePort* peer) 
    : SimPortBase(module, false) 
  {
    this->connect(peer);
  }

  void send(const Pkt& pkt, uint64_t delay) const;

  const Func& func() const {
    return func_;
  }

protected:
  SlavePort& operator=(const SlavePort&);
  Func func_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Pkt>
class MasterPort : public SimPortBase {
public:
  typedef std::shared_ptr<MasterPort<Ptr>> Ptr;
  typedef std::function<void (const Pkt&, uint32_t)> Func;

  static Ptr Create() {
    return std::make_shared<MasterPort<Ptr>>(module);
  }  

  MasterPort(SimObjectBase* module) : SimPortBase(module, false) {}

  MasterPort(SimObjectBase* module, MasterPort* peer) 
    : SimPortBase(module, false) 
  {
    peer->connect(this);
  }

  void bind(SlavePort<Pkt>* peer) {
    this->connect(peer);
  }

  void unbind() {    
    peer_->disconnect();
    this->disconnect();
  }

  void send(const Pkt& pkt, uint64_t delay) const {
    assert(peer_ != nullptr);
    if (peer_->is_slave()) {
      auto slave = reinterpret_cast<const SlavePort<Pkt>*>(peer_);
      slave->send(pkt, delay);
    } else {
      auto master = reinterpret_cast<const MasterPort<Pkt>*>(peer_);
      master->send(pkt, delay);
    }  
  }

private:
  MasterPort& operator=(const MasterPort&);
};

///////////////////////////////////////////////////////////////////////////////

class SimContext;

class SimObjectBase {
public:
  typedef std::shared_ptr<SimObjectBase> Ptr;

  virtual ~SimObjectBase() {}

  template <typename T, typename Pkt>
  void schedule(T *obj, void (T::*entry)(const Pkt&), const Pkt& pkt, uint64_t delay);

  virtual void step(uint64_t cycle) = 0;

  const std::string& name() const {
    return name_;
  }

protected:

  SimObjectBase(const SimContext& ctx, const char* name);

  uint32_t allocate_port(SimPortBase* port) {
      uint32_t id = ports_.size();
      ports_.push_back(port);
      return id;
  }

private:
  std::string name_;
  std::vector<SimPortBase*> ports_;

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

  void finalize() {
    instance().clear();
  }

  void register_object(const SimObjectBase::Ptr& obj) {
    objects_.push_back(obj);
  }

  template <typename Pkt>
  void schedule(const typename SimSimpleEvent<Pkt>::Func& callback, 
                const Pkt& pkt, 
                uint64_t delay) {    
    auto evt = SimSimpleEvent<Pkt>::Create(callback, pkt, delay);
    assert(delay != 0);
    events_.emplace_back(evt);
  }

  template <typename Pkt>
  void schedule(const typename SimPortEvent<Pkt>::Func& callback, 
                const Pkt& pkt, 
                uint32_t port_id, 
                uint64_t delay) {
    auto evt = SimPortEvent<Pkt>::Create(callback, pkt, port_id, delay);
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

inline SimPortBase::SimPortBase(SimObjectBase* module, bool is_slave) 
  : module_(module)  
  , port_id_(module->allocate_port(this))
  , is_slave_(is_slave)
  , peer_(nullptr) 
{}

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
void SlavePort<Pkt>::send(const Pkt& pkt, uint64_t delay) const {
  if (func_) {
    SimPlatform::instance().schedule(func_, pkt, port_id_, delay);
  } else {
    assert(peer_ != nullptr);
    if (peer_->is_slave()) {
      auto slave = reinterpret_cast<const SlavePort<Pkt>*>(peer_);
      slave->send(pkt, delay);
    } else {
      auto master = reinterpret_cast<const MasterPort<Pkt>*>(peer_);
      master->send(pkt, delay);
    }
  }  
}

template <typename T, typename Pkt>
void SimObjectBase::schedule(T *obj, void (T::*entry)(const Pkt&), const Pkt& pkt, uint64_t delay) {
  auto callback = std::bind(entry, obj, std::placeholders::_1);
  SimPlatform::instance().schedule(callback, pkt, delay);
}

}