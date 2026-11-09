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

#include <cstdint>
#include <deque>
#include <optional>

namespace vortex {

// Timing of one AXI4 memory port as a latency-bandwidth pipe: a fixed latency
// per transaction and a bound on the reads and writes in flight, one beat per
// cycle on each response channel. Cycle for cycle this is FireSim FASED's
// "Latency Bandwidth Pipe" (LatencyBandwidthPipe.scala over TimingModel.scala's
// SplitTransactionModel and AXI4Releaser); the defaults are FASED's hardware
// defaults. Every output is a function of registered state only, so the caller
// samples the port's request side once per cycle, before the clock edge, and
// calls step() after it. Timing only: responses leave in the order their
// requests were accepted, so the caller keeps the data alongside.
class AxiLatencyPipe {
public:
  struct Config {
    uint32_t read_latency  = 30;
    uint32_t write_latency = 30;
    uint32_t max_reads     = 16;
    uint32_t max_writes    = 16;
  };

  struct Inputs {
    bool     ar_valid = false;
    uint32_t ar_id    = 0;
    uint32_t ar_len   = 0;
    bool     aw_valid = false;
    uint32_t aw_id    = 0;
    bool     w_valid  = false;
    bool     w_last   = false;
    bool     r_ready  = false;
    bool     b_ready  = false;
  };

  AxiLatencyPipe() : cfg_() {}
  explicit AxiLatencyPipe(const Config& cfg) : cfg_(cfg) {}

  void reset() {
    *this = AxiLatencyPipe(cfg_);
  }

  bool ar_ready() const { return pending_reads_ < cfg_.max_reads; }
  bool aw_ready() const { return pending_aw_ < cfg_.max_writes; }
  bool w_ready()  const { return pending_w_ < cfg_.max_writes; }

  bool     r_valid()   const { return cur_read_.has_value(); }
  uint32_t r_id()      const { return cur_read_->id; }
  bool     r_last()    const { return cur_read_->beat == cur_read_->len; }
  uint32_t r_beat()    const { return cur_read_->beat; }

  bool     b_valid() const { return cur_write_.has_value(); }
  uint32_t b_id()    const { return cur_write_->id; }

  // Advances one cycle on the request-side values sampled before the edge.
  void step(const Inputs& in) {
    bool ar_fire    = in.ar_valid && this->ar_ready();
    bool aw_fire    = in.aw_valid && this->aw_ready();
    bool wlast_fire = in.w_valid && in.w_last && this->w_ready();
    bool r_fire     = this->r_valid() && in.r_ready;
    bool rlast_fire = r_fire && this->r_last();
    bool b_fire     = this->b_valid() && in.b_ready;

    // Reads: latency pipe (a flow queue) into a one-entry pipe stage that
    // drives R. Release is one cycle early, since that stage adds the cycle.
    {
      std::optional<Read> arrived;
      if (ar_fire) {
        arrived = Read{cycle_ + cfg_.read_latency - 1, in.ar_id, in.ar_len, 0};
      }
      Read* head = !read_pipe_.empty() ? &read_pipe_.front() : (arrived ? &*arrived : nullptr);
      bool release = head && head->release <= cycle_;
      bool stage_ready = !cur_read_ || rlast_fire;
      if (r_fire && !rlast_fire) {
        ++cur_read_->beat;
      } else if (rlast_fire) {
        cur_read_.reset();
      }
      if (release && stage_ready) {
        cur_read_ = *head;
        if (!read_pipe_.empty()) {
          read_pipe_.pop_front();
        } else {
          arrived.reset();
        }
      }
      if (arrived) {
        read_pipe_.push_back(*arrived);
      }
    }

    // Writes: an AW and its last W beat, in either order, form one write; its
    // latency counts from the cycle both have arrived.
    {
      bool new_write = (pending_w_ > pending_aw_ && aw_fire)
                    || (pending_w_ < pending_aw_ && wlast_fire)
                    || (aw_fire && wlast_fire);
      std::optional<uint32_t> aw_arrived;
      if (aw_fire) {
        aw_arrived = in.aw_id;
      }
      std::optional<Write> formed;
      if (new_write) {
        uint32_t id;
        if (!aw_queue_.empty()) {
          id = aw_queue_.front();
          aw_queue_.pop_front();
        } else {
          id = *aw_arrived;
          aw_arrived.reset();
        }
        formed = Write{cycle_ + cfg_.write_latency - 1, id};
      }
      if (aw_arrived) {
        aw_queue_.push_back(*aw_arrived);
      }
      Write* head = !write_pipe_.empty() ? &write_pipe_.front() : (formed ? &*formed : nullptr);
      bool release = head && head->release <= cycle_;
      bool stage_ready = !cur_write_ || b_fire;
      if (b_fire) {
        cur_write_.reset();
      }
      if (release && stage_ready) {
        cur_write_ = *head;
        if (!write_pipe_.empty()) {
          write_pipe_.pop_front();
        } else {
          formed.reset();
        }
      }
      if (formed) {
        write_pipe_.push_back(*formed);
      }
    }

    pending_reads_ = count(pending_reads_, ar_fire, rlast_fire);
    pending_aw_    = count(pending_aw_, aw_fire, b_fire);
    pending_w_     = count(pending_w_, wlast_fire, b_fire);
    ++cycle_;
  }

private:
  struct Read {
    uint64_t release;
    uint32_t id;
    uint32_t len;
    uint32_t beat;
  };

  struct Write {
    uint64_t release;
    uint32_t id;
  };

  static uint32_t count(uint32_t value, bool inc, bool dec) {
    if (inc && !dec) return value + 1;
    if (dec && !inc && value != 0) return value - 1;
    return value;
  }

  Config cfg_;
  uint64_t cycle_ = 0;
  uint32_t pending_reads_ = 0;
  uint32_t pending_aw_ = 0;
  uint32_t pending_w_ = 0;
  std::deque<Read> read_pipe_;
  std::optional<Read> cur_read_;
  std::deque<uint32_t> aw_queue_;
  std::deque<Write> write_pipe_;
  std::optional<Write> cur_write_;
};

} // namespace vortex
