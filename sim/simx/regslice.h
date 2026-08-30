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
#include <cstdint>
#include "types.h"

// Kernel-agnostic framework element: compiles against whichever simulation
// kernel types.h selected.
namespace vortex {

// Registered boundary stage: a one-entry-per-tick forwarder whose outgoing
// send carries an explicit >=1 cycle delay. Placed at an execution-domain
// boundary and owned by the SENDING domain, it guarantees the cross-domain
// edge is never combinational regardless of the arbiter chain behind it.
// Backpressure is credit-based: the stall decision reads only the local
// credit counter, never the downstream endpoint's occupancy; each downstream
// pop returns a credit one cycle later (a registered ready, like a real
// pipeline-boundary crossing).
template <typename Type>
class RegSlice : public SimObject<RegSlice<Type>> {
public:
  using Ptr = std::shared_ptr<RegSlice<Type>>;

  SimChannel<Type> In;
  SimChannel<Type> Out;

  RegSlice(const SimContext& ctx, const char* name, uint32_t delay)
    : SimObject<RegSlice<Type>>(ctx, name)
    , In(this)
    , Out(this)
    , delay_(delay)
    , credits_(-1) {
    __assert(delay >= 1, "boundary stage requires a registered delay");
    Out.mark_boundary();
  }

protected:
  void on_reset() {
    // Runs single-threaded (workers parked): safe point to (re)install the
    // credit-return hook on the downstream endpoint and refill the pool —
    // in-flight credit returns were dropped by the platform reset.
    this->init_credits();
  }

  void on_tick() {
    __assert(credits_ >= 0, "boundary credit underflow");
    if (!In.empty() && credits_ > 0) {
      Out.send(In.peek(), delay_);
      --credits_;
      In.pop();
    }
    if (In.size() == 0) {
      this->tick_sleep();
    }
  }

private:
  // Runs at reset, after bind(): the downstream endpoint and its capacity
  // exist by then. Credits track downstream slots exactly (send consumes,
  // pop returns after one cycle), so Out.send can never overflow the sink.
  void init_credits() {
    SimChannelBase* endpoint = &Out;
    while (endpoint->sink()) {
      endpoint = endpoint->sink();
    }
    // A full credit refill assumes no stale packets hold downstream slots.
    __assert(endpoint->size() == 0, "boundary refill with stale downstream packets");
    credits_ = static_cast<int32_t>(Out.capacity());
    endpoint->pop_callback([this]() {
      // Credit returns to the producing domain with one cycle of latency
      // (a registered ready), delivered by the producer's own thread.
      SimPlatform::instance().cross_call(this->domain(), [this]() {
        ++credits_;
        this->tick_wake();
      });
    });
  }

  uint32_t delay_;
  int32_t credits_;

  friend class SimObject<RegSlice<Type>>;
};

} // namespace vortex
