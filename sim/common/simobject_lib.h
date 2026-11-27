// Copyright © 2026  Vortex GPGPU
// SPDX-License-Identifier: MIT
//
// simobject_lib.h: library units and helpers on simobject.h: elastic and
// delay buffers, the boundary stage, arbiters and crossbars with response
// routing, occupancy statistics and arbitration policies.
//
// A unit's own queue needs no class: it is a SimPortIn<T> port the unit fills
// with in.push(pkt, delay), and a queue between two units is the reader's
// port, bound with writer.bind(&port, delay).

#pragma once

#include <functional>

#include "simobject.h"

namespace vortex {

/// An occupancy integral: the sum over cycles of a value, as held at the end
/// of each cycle. Replaces `latency += outstanding` on every tick, which kept
/// the counting unit awake. A change costs one multiply-add.
class Occupancy {
public:
  /// Change the value from the current cycle on.
  void add(int64_t delta) {
    uint64_t now = SimPlatform::instance().cycles();
    sum_ += value_ * (now - since_);
    since_ = now;
    value_ = uint64_t(int64_t(value_) + delta);
  }

  uint64_t value() const { return value_; }

  /// The integral over every cycle so far, the current one included.
  uint64_t total() const {
    return sum_ + value_ * (SimPlatform::instance().cycles() + 1 - since_);
  }

private:
  uint64_t sum_ = 0;
  uint64_t value_ = 0;
  uint64_t since_ = 0;
};

enum class ArbiterType : uint8_t {
  Priority,
  RoundRobin,
  Matrix,
  GreedyRR,
  GreedyMatrix
};

/// An arbiter over up to 64 requesters, requests as a mask. select() names
/// the requester it would grant without changing its state; commit() records
/// that grant, so a grant the receiver refuses leaves the order as it was.
/// Matrix is least-recently-granted, as VX_matrix_arbiter: a grant drops its
/// requester to the lowest priority; from reset the highest index ranks
/// first. A greedy arbiter keeps granting its owner while it requests.
class Arbiter {
public:
  static constexpr uint32_t kNone = ~uint32_t(0);

  Arbiter(ArbiterType type = ArbiterType::Priority, uint32_t size = 0)
    : type_(type)
    , size_(size) {
    assert(size <= 64);
    this->reset();
  }

  void reset() {
    last_ = (type_ == ArbiterType::RoundRobin) ? 0 : kNone;
    for (uint32_t k = 0; k < size_; ++k) {
      order_[k] = uint8_t(size_ - 1 - k);
    }
  }

  /// The granted requester. Requires requests != 0.
  uint32_t select(uint64_t requests) const {
    switch (type_) {
    case ArbiterType::Priority:
      return uint32_t(__builtin_ctzll(requests));
    case ArbiterType::RoundRobin:
      return rotate_first(requests, (last_ + 1) % size_);
    case ArbiterType::GreedyRR:
      if (last_ < size_ && ((requests >> last_) & 1)) {
        return last_;
      }
      return rotate_first(requests, (last_ < size_) ? (last_ + 1) % size_ : 0);
    case ArbiterType::Matrix:
    case ArbiterType::GreedyMatrix:
      if (type_ == ArbiterType::GreedyMatrix && last_ < size_ && ((requests >> last_) & 1)) {
        return last_;
      }
      for (uint32_t k = 0; k < size_; ++k) {
        if ((requests >> order_[k]) & 1) {
          return order_[k];
        }
      }
      return kNone;
    }
    return kNone;
  }

  /// The granted requester of a bit set (`test(i)`, `size()`); kNone when
  /// none requests.
  template <typename Bits, typename = decltype(std::declval<const Bits&>().test(0))>
  uint32_t select(const Bits& requests) const {
    assert(requests.size() == size_);
    uint64_t mask = 0;
    for (uint32_t i = 0; i < size_; ++i) {
      mask |= uint64_t(requests.test(i)) << i;
    }
    return (mask != 0) ? this->select(mask) : kNone;
  }

  /// Record the grant of `idx` (call when the grant is accepted).
  void commit(uint32_t idx) {
    switch (type_) {
    case ArbiterType::Priority:
      break;
    case ArbiterType::RoundRobin:
    case ArbiterType::GreedyRR:
      last_ = idx;
      break;
    case ArbiterType::Matrix:
    case ArbiterType::GreedyMatrix: {
      if (type_ == ArbiterType::GreedyMatrix && idx == last_) {
        break;
      }
      uint32_t k = 0;
      while (order_[k] != idx) {
        ++k;
      }
      for (; k + 1 < size_; ++k) {
        order_[k] = order_[k + 1];
      }
      order_[size_ - 1] = uint8_t(idx);
      last_ = idx;
      break;
    }
    }
  }

private:
  // The first set bit at or after `start`, wrapping around.
  static uint32_t rotate_first(uint64_t requests, uint32_t start) {
    uint64_t high = requests & (~uint64_t(0) << start);
    return uint32_t(__builtin_ctzll(high != 0 ? high : requests));
  }

  ArbiterType type_;
  uint32_t size_;
  uint32_t last_ = 0;
  uint8_t order_[64] = {};
};

/// The default delay of the library units: a register when either side is
/// wider than 2.
inline uint32_t default_delay(uint32_t num_inputs, uint32_t num_outputs) {
  return (num_inputs > 2 || num_outputs > 2) ? 1 : 0;
}

/// A registered elastic buffer: In holds its entries, and the head moves on
/// through Out one register later, one per cycle.
template <typename Type>
class TxBuffer : public SimObject<TxBuffer<Type>> {
public:
  SimPortIn<Type> In;
  SimPortOut<Type> Out;

  TxBuffer(const SimContext& ctx, const char* name, uint32_t size)
    : SimObject<TxBuffer<Type>>(ctx, name)
    , In(this, size, &TxBuffer::forward)
    , Out(this) {}

private:
  bool forward(Type& pkt) {
    if (Out.full()) {
      return false;
    }
    Out.send(pkt, 1);
    return true;
  }
};

/// An elastic buffer with a fixed latency: a packet that reaches In on cycle
/// t leaves through Out to arrive on cycle t + latency, one per cycle at full
/// throughput. In holds the packets in flight, so its capacity is the
/// buffer's size, and a full buffer back-pressures the producer.
template <typename Type>
class DelayBuffer : public SimObject<DelayBuffer<Type>> {
public:
  SimPortIn<Type> In;
  SimPortOut<Type> Out;

  DelayBuffer(const SimContext& ctx, const char* name, uint32_t latency)
    : SimObject<DelayBuffer<Type>>(ctx, name)
    , In(this, size_for(latency))
    , Out(this)
    , latency_(latency) {
    assert(latency >= 1 && "a delay buffer needs a latency");
  }

  // A FIFO for the packets in flight at full throughput plus the one
  // arriving, then a two-entry output stage.
  static uint32_t size_for(uint32_t latency) {
    return ((latency > 1) ? (1u << log2ceil(latency)) : 0) + 2;
  }

protected:
  bool busy() const { return false; }  // no work beyond its inputs

  void on_tick() {
    // Sent with a one-cycle delay, so the head leaves a cycle early.
    uint64_t now = SimPlatform::instance().cycles();
    if (!In.empty() && In.head_cycle() + latency_ - 1 <= now && !Out.full()) {
      Out.send(In.pop(), 1);
    }
  }

private:
  uint32_t latency_;

  friend class SimObject<DelayBuffer<Type>>;
};

/// A registered boundary stage: one packet per cycle from In to Out, sent
/// with an explicit delay of at least one cycle. Placed at an execution-domain
/// boundary and owned by the sending domain, it keeps the crossing from being
/// combinational. Back-pressure is credit-based: the stall decision reads only
/// the local credit count, never the downstream port's occupancy, and each
/// downstream pop returns a credit one cycle later (a registered ready).
template <typename Type>
class RegSlice : public SimObject<RegSlice<Type>> {
public:
  SimPortIn<Type> In;
  SimPortOut<Type> Out;

  RegSlice(const SimContext& ctx, const char* name, uint32_t delay)
    : SimObject<RegSlice<Type>>(ctx, name)
    , In(this, 2)
    , Out(this)
    , delay_(delay) {
    assert(delay >= 1 && "a boundary stage needs a registered delay");
    Out.mark_boundary();
  }

protected:
  // Runs single-threaded, after binding: the downstream port and its
  // capacity exist, and the platform reset dropped in-flight credit returns.
  void on_reset() {
    SimPortInBase* endpoint = Out.endpoint();
    // A full credit refill assumes no stale packets hold downstream slots.
    assert(endpoint->size() == 0 && "boundary refill with stale downstream packets");
    credits_ = static_cast<int32_t>(Out.capacity());
    endpoint->pop_callback([this]() {
      // The credit returns to the producing domain one cycle later, delivered
      // by the producer's own thread.
      SimPlatform::instance().run_in_domain(this->domain(), [this]() {
        ++credits_;
        this->tick_wake();
      });
    });
  }

  bool busy() const { return false; }  // no work beyond its inputs

  void on_tick() {
    assert(credits_ >= 0 && "boundary credit underflow");
    if (!In.empty() && credits_ > 0) {
      Out.send(In.pop(), delay_);
      --credits_;
    }
  }

private:
  uint32_t delay_;
  int32_t credits_ = -1;

  friend class SimObject<RegSlice<Type>>;
};

/// Arbitrates groups of inputs onto outputs: output o serves inputs
/// [o*R, o*R + R), R = the next power of two of ceil(inputs / outputs), one
/// packet per output per cycle. Each output packet carries the input it came
/// from.
template <typename Type>
class TxArbiter : public SimObject<TxArbiter<Type>> {
public:
  typedef Type ReqType;

  struct RspType {
    Type     data;
    uint32_t input;

    RspType(const Type& _data, uint32_t _input = 0)
      : data(_data)
      , input(_input)
    {}

    RspType(Type&& _data, uint32_t _input = 0)
      : data(std::move(_data))
      , input(_input)
    {}

    operator Type() const {
      return data;
    }
  };

  SimPortInArray<ReqType> Inputs;
  SimPortOutArray<RspType> Outputs;

  TxArbiter(const SimContext& ctx,
            const char* name,
            ArbiterType type,
            uint32_t num_inputs,
            uint32_t num_outputs,
            uint32_t delay)
    : SimObject<TxArbiter<Type>>(ctx, name)
    , Inputs(this, num_inputs, 2)
    , Outputs(this, num_outputs)
    , delay_(delay)
    , group_(1u << log2ceil((num_inputs + num_outputs - 1) / num_outputs))
    , arbiters_(num_outputs, Arbiter(type, group_)) {
    assert(num_inputs <= 64 && num_outputs <= 64 && num_inputs >= num_outputs);
    // bypass mode
    if (num_inputs == num_outputs) {
      for (uint32_t i = 0; i < num_inputs; ++i) {
        Inputs[i].bypass_to(Outputs[i], [](Type&& pkt) { return RspType(std::move(pkt)); });
      }
    }
  }

  TxArbiter(const SimContext& ctx,
            const char* name,
            ArbiterType type,
            uint32_t num_inputs,
            uint32_t num_outputs)
    : TxArbiter(ctx, name, type, num_inputs, num_outputs, default_delay(num_inputs, num_outputs)) {}

protected:
  void on_reset() {
    for (auto& arbiter : arbiters_) {
      arbiter.reset();
    }
  }

  bool busy() const { return false; }  // no work beyond its inputs

  void on_tick() {
    if (Inputs[0].bypassed()) {
      return;
    }
    for (uint32_t o = 0; o < Outputs.size(); ++o) {
      uint64_t requests = Inputs.ready_mask(o * group_, group_);
      if (requests == 0 || Outputs[o].full()) {
        continue;
      }
      uint32_t g = arbiters_[o].select(requests);
      uint32_t i = o * group_ + g;
      Outputs[o].send(RspType(Inputs[i].pop(), i), delay_);
      arbiters_[o].commit(g);
    }
  }

private:
  uint32_t delay_;
  uint32_t group_;
  std::vector<Arbiter> arbiters_;

  friend class SimObject<TxArbiter<Type>>;
};

/// Routes each input to the output `output_sel(pkt)` names; each output takes
/// one packet per cycle, round robin or priority among the inputs routed to
/// it. Each output packet carries the input it came from.
template <typename Type>
class TxCrossBar : public SimObject<TxCrossBar<Type>> {
public:
  typedef Type ReqType;

  struct RspType {
    Type     data;
    uint32_t input;

    RspType(const Type& _data, uint32_t _input = 0)
      : data(_data)
      , input(_input)
    {}

    RspType(Type&& _data, uint32_t _input = 0)
      : data(std::move(_data))
      , input(_input)
    {}

    operator Type() const {
      return data;
    }
  };

  SimPortInArray<ReqType> Inputs;
  SimPortOutArray<RspType> Outputs;

  TxCrossBar(const SimContext& ctx,
             const char* name,
             ArbiterType type,
             uint32_t num_inputs,
             uint32_t num_outputs,
             std::function<uint32_t(const Type& req)> output_sel,
             uint32_t delay)
    : SimObject<TxCrossBar<Type>>(ctx, name)
    , Inputs(this, num_inputs, 2)
    , Outputs(this, num_outputs)
    , round_robin_(type == ArbiterType::RoundRobin)
    , delay_(delay)
    , last_grant_(num_outputs, num_inputs - 1)
    , dest_(num_inputs)
    , output_sel_(std::move(output_sel)) {
    assert(num_outputs <= 64 && ispow2(num_outputs));
    assert(output_sel_ != nullptr);
    assert(type == ArbiterType::Priority || type == ArbiterType::RoundRobin);
    // bypass mode
    if (num_inputs == 1 && num_outputs == 1) {
      Inputs[0].bypass_to(Outputs[0], [](Type&& pkt) { return RspType(std::move(pkt)); });
    }
  }

  TxCrossBar(const SimContext& ctx,
             const char* name,
             ArbiterType type,
             uint32_t num_inputs,
             uint32_t num_outputs,
             std::function<uint32_t(const Type& req)> output_sel)
    : TxCrossBar(ctx, name, type, num_inputs, num_outputs, std::move(output_sel),
                 default_delay(num_inputs, num_outputs)) {}

  uint64_t collisions() const { return collisions_; }

protected:
  void on_reset() {
    std::fill(last_grant_.begin(), last_grant_.end(), Inputs.size() - 1);
  }

  bool busy() const { return false; }  // no work beyond its inputs

  void on_tick() {
    if (Inputs[0].bypassed()) {
      return;
    }
    uint32_t num_in = Inputs.size();
    uint32_t num_out = Outputs.size();
    // Each ready input's destination, computed once per head.
    bool any = false;
    for (uint32_t i = 0; i < num_in; ++i) {
      dest_[i] = this->route(i);
      any |= (dest_[i] != kIdle);
    }
    if (!any) {
      return;
    }
    for (uint32_t o = 0; o < num_out; ++o) {
      uint32_t start = round_robin_ ? last_grant_[o] + 1 : 0;
      uint32_t winner = kIdle;
      bool collision = false;
      for (uint32_t k = 0; k < num_in; ++k) {
        uint32_t i = (start + k) % num_in;
        if (dest_[i] != o) {
          continue;
        }
        if (winner != kIdle) {
          collision = true;
          break;
        }
        winner = i;
      }
      if (winner == kIdle) {
        continue;
      }
      collisions_ += collision;
      if (Outputs[o].full()) {
        continue;
      }
      Outputs[o].send(RspType(Inputs[winner].pop(), winner), delay_);
      last_grant_[o] = winner;
      // The input's next packet, if ready, competes for the later outputs in
      // this same cycle.
      dest_[winner] = this->route(winner);
    }
  }

private:
  static constexpr uint32_t kIdle = ~uint32_t(0);

  uint32_t route(uint32_t i) {
    if (Inputs[i].empty()) {
      return kIdle;
    }
    return (Outputs.size() == 1) ? 0 : output_sel_(Inputs[i].peek());
  }

  bool round_robin_;
  uint32_t delay_;
  std::vector<uint32_t> last_grant_;
  std::vector<uint32_t> dest_;
  std::function<uint32_t(const Type& req)> output_sel_;
  uint64_t collisions_ = 0;

  friend class SimObject<TxCrossBar<Type>>;
};

/// Request arbiter with response routing: requests from group r of output o
/// leave with `tag = (tag << log2 R) | r`; a response on output o returns to
/// input o*R + (tag & (R-1)) with the low bits stripped. Req and Rsp have a
/// `tag` field.
template <typename Req, typename Rsp>
class TxRxArbiter : public SimObject<TxRxArbiter<Req, Rsp>> {
public:
  typedef Req ReqType;
  typedef Rsp RspType;

  SimPortInArray<Req>  ReqIn;
  SimPortOutArray<Rsp> RspOut;

  SimPortOutArray<Req> ReqOut;
  SimPortInArray<Rsp>  RspIn;

  TxRxArbiter(const SimContext& ctx,
              const char* name,
              ArbiterType type,
              uint32_t num_inputs,
              uint32_t num_outputs,
              uint32_t req_delay,
              uint32_t rsp_delay)
    : SimObject<TxRxArbiter<Req, Rsp>>(ctx, name)
    , ReqIn(this, num_inputs, 2)
    , RspOut(this, num_inputs)
    , ReqOut(this, num_outputs)
    , RspIn(this, num_outputs, 2)
    , req_delay_(req_delay)
    , rsp_delay_(rsp_delay)
    , lg2_num_reqs_(log2ceil((num_inputs + num_outputs - 1) / num_outputs))
    , arbiters_(num_outputs, Arbiter(type, 1u << lg2_num_reqs_)) {
    assert(num_inputs <= 64 && num_outputs <= 64 && num_inputs >= num_outputs);
    // bypass mode
    if (num_inputs == num_outputs) {
      for (uint32_t i = 0; i < num_inputs; ++i) {
        ReqIn[i].bypass_to(ReqOut[i]);
        RspIn[i].bypass_to(RspOut[i]);
      }
    }
  }

  TxRxArbiter(const SimContext& ctx,
              const char* name,
              ArbiterType type,
              uint32_t num_inputs,
              uint32_t num_outputs)
    : TxRxArbiter(ctx, name, type, num_inputs, num_outputs,
                  default_delay(num_inputs, num_outputs), default_delay(num_inputs, num_outputs)) {}

protected:
  void on_reset() {
    for (auto& arbiter : arbiters_) {
      arbiter.reset();
    }
  }

  bool busy() const { return false; }  // no work beyond its inputs

  void on_tick() {
    if (ReqIn[0].bypassed()) {
      return;
    }
    uint32_t R = 1u << lg2_num_reqs_;
    for (uint32_t o = 0; o < ReqOut.size(); ++o) {
      uint64_t requests = ReqIn.ready_mask(o * R, R);
      if (requests == 0 || ReqOut[o].full()) {
        continue;
      }
      uint32_t g = arbiters_[o].select(requests);
      Req req = ReqIn[o * R + g].pop();
      if (lg2_num_reqs_ != 0) {
        req.tag = (req.tag << lg2_num_reqs_) | g;
      }
      ReqOut[o].send(std::move(req), req_delay_);
      arbiters_[o].commit(g);
    }
    for (uint32_t o = 0; o < RspIn.size(); ++o) {
      if (RspIn[o].empty()) {
        continue;
      }
      uint32_t r = (lg2_num_reqs_ != 0) ? (RspIn[o].peek().tag & (R - 1)) : 0;
      auto& rsp_out = RspOut[o * R + r];
      if (rsp_out.full()) {
        continue;
      }
      Rsp rsp = RspIn[o].pop();
      rsp.tag >>= lg2_num_reqs_;
      rsp_out.send(std::move(rsp), rsp_delay_);
    }
  }

private:
  uint32_t req_delay_;
  uint32_t rsp_delay_;
  uint32_t lg2_num_reqs_;
  std::vector<Arbiter> arbiters_;

  friend class SimObject<TxRxArbiter<Req, Rsp>>;
};

/// Request crossbar with response routing: a request from input i leaves
/// with `tag = (tag << log2 I) | i`; each input takes one response per cycle,
/// chosen among the outputs holding one for it by a single shared arbiter.
template <typename Req, typename Rsp>
class TxRxCrossBar : public SimObject<TxRxCrossBar<Req, Rsp>> {
  typedef TxCrossBar<Req> ReqXbar;

  // The request side: a crossbar unit of its own, ahead of this one in the
  // tick order, whose outputs reach ReqOut through tagging wires.
  typename ReqXbar::Ptr crossbar_;

public:
  typedef Req ReqType;
  typedef Rsp RspType;

  SimPortInArray<Req>& ReqIn;
  SimPortOutArray<Rsp> RspOut;

  SimPortOutArray<Req> ReqOut;
  SimPortInArray<Rsp>  RspIn;

  TxRxCrossBar(const SimContext& ctx,
               const char* name,
               ArbiterType type,
               uint32_t num_inputs,
               uint32_t num_outputs,
               std::function<uint32_t(const Req& req)> output_sel,
               uint32_t req_delay,
               uint32_t rsp_delay)
    : SimObject<TxRxCrossBar<Req, Rsp>>(ctx, name)
    , crossbar_(ReqXbar::Create(name, type, num_inputs, num_outputs, std::move(output_sel), req_delay))
    , ReqIn(crossbar_->Inputs)
    , RspOut(this, num_inputs)
    , ReqOut(this, num_outputs)
    , RspIn(this, num_outputs, 2)
    , req_tagged_(this, num_outputs, 2)
    , arbiter_(type, num_outputs)
    , rsp_delay_(rsp_delay)
    , lg2_inputs_(log2ceil(num_inputs)) {
    for (uint32_t o = 0; o < num_outputs; ++o) {
      crossbar_->Outputs[o].bind(&req_tagged_[o]);
      req_tagged_[o].bypass_to(ReqOut[o], [lg2_inputs = lg2_inputs_](typename ReqXbar::RspType&& xbar_rsp) {
        Req req(std::move(xbar_rsp.data));
        if (lg2_inputs != 0) {
          req.tag = (req.tag << lg2_inputs) | xbar_rsp.input;
        }
        return req;
      });
    }
    // bypass mode
    if (num_inputs == 1 && num_outputs == 1) {
      RspIn[0].bypass_to(RspOut[0]);
    }
  }

  TxRxCrossBar(const SimContext& ctx,
               const char* name,
               ArbiterType type,
               uint32_t num_inputs,
               uint32_t num_outputs,
               std::function<uint32_t(const Req& req)> output_sel)
    : TxRxCrossBar(ctx, name, type, num_inputs, num_outputs, std::move(output_sel),
                   default_delay(num_inputs, num_outputs), default_delay(num_inputs, num_outputs)) {}

  uint64_t collisions() const { return crossbar_->collisions(); }

protected:
  void on_reset() { arbiter_.reset(); }

  bool busy() const { return false; }  // no work beyond its inputs

  void on_tick() {
    if (RspIn[0].bypassed()) {
      return;
    }
    uint32_t mask = (1u << lg2_inputs_) - 1;
    for (uint32_t i = 0; i < RspOut.size(); ++i) {
      uint64_t requests = 0;
      for (uint32_t o = 0; o < RspIn.size(); ++o) {
        if (!RspIn[o].empty() && (lg2_inputs_ == 0 || (RspIn[o].peek().tag & mask) == i)) {
          requests |= uint64_t(1) << o;
        }
      }
      if (requests == 0) {
        continue;
      }
      uint32_t g = arbiter_.select(requests);
      if (RspOut[i].full()) {
        continue;
      }
      Rsp rsp = RspIn[g].pop();
      rsp.tag >>= lg2_inputs_;
      RspOut[i].send(std::move(rsp), rsp_delay_);
      arbiter_.commit(g);
    }
  }

private:
  SimPortInArray<typename ReqXbar::RspType> req_tagged_;
  Arbiter arbiter_;
  uint32_t rsp_delay_;
  uint32_t lg2_inputs_;

  friend class SimObject<TxRxCrossBar<Req, Rsp>>;
};

}  // namespace vortex
