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
#include <vector>
#include "types.h"
#include "VX_types.h"

namespace vortex {

struct kmu_req_t {
  uint64_t PC;
  uint64_t entry;
  uint64_t param;
  uint32_t cta_id;
  uint32_t block_idx[3];
  uint32_t block_dim[3];
  uint32_t grid_dim[3];
  uint32_t lmem_size;
  uint32_t block_size;
  uint32_t warp_step[3];
  uint32_t cluster_dim[3];
  bool     is_first_of_cluster;
  // Launch identity: bumped on every start(), so consecutive launches of the
  // same kernel are still distinct contexts.
  uint8_t  ctx_id;
};

// Kernel Management Unit. Holds the in-flight kernel descriptor, walks the
// grid, and pushes one CTA per cycle onto a ready core's launch lane.
class Kmu : public SimObject<Kmu> {
public:
  // Per-core launch lanes. Each lane is bound to a registered boundary stage
  // feeding the core dispatcher's endpoint; readiness is the lane's local
  // full() view — the KMU never reads a consumer's state.
  std::vector<SimChannel<kmu_req_t>> bus_out;

  // Per-core credit returns: the dispatcher sends one message per admitted
  // CTA. Dispatch is additionally gated on the in-flight count so pacing
  // rests on explicit slot accounting, not on transport back-pressure.
  std::vector<SimChannel<uint8_t>> credit_in;

  // In-flight bound per core: the boundary stage's endpoint (2) plus the
  // dispatcher endpoint (1) plus one admission credit in return flight, so
  // the lane's full() view stays the binding gate whenever the transport
  // applies back-pressure.
  static constexpr uint32_t LAUNCH_CREDITS = 4;

  Kmu(const SimContext& ctx, const char* name, uint32_t num_cores);

  void dcr_write(uint32_t addr, uint32_t value);

  // Called by ProcessorImpl::run() to arm a kernel launch (host context,
  // between cycles).
  void start();

  // True while CTAs remain to be issued.
  bool running() const { return running_; }

  // A grid-less launch is a delegated draw launch: the KMU walks no CTAs and
  // the processor forwards the frame kick to the raster engines.
  bool launch_delegated() const {
    return !(grid_dim_[0] && grid_dim_[1] && grid_dim_[2]);
  }

  // Program image base PC (KMU STARTUP_ADDR) — where every warp begins
  // executing (__vx_cta_entry). Persists across reset; the Fragment Work
  // Distributor reuses it as the launch PC for injected fragment warps.
  uint64_t startup_pc() const { return PC_; }

protected:
  void on_reset();
  void on_tick();

private:
  // Fill *req with the next CTA for `core_id`; false when the grid is
  // exhausted. The caller steers: a locked cluster's members all target the
  // owning core.
  bool next_cta(kmu_req_t* req, uint32_t core_id);

  uint64_t PC_;
  uint64_t entry_;
  uint64_t param_;
  uint32_t block_dim_[3];
  uint32_t grid_dim_[3];
  uint32_t lmem_size_;
  uint32_t block_size_;
  uint32_t warp_step_[3];
  uint32_t cluster_dim_[3];
  bool     running_;
  uint32_t cta_id_;
  // Round-robin pointer over launch lanes; advances past the granted core
  // when a message (cluster or standalone CTA) completes.
  uint32_t rr_ptr_;
  uint8_t  ctx_id_;
  // Nested counter: group_origin advances by cluster_dim per axis;
  // intra_offset walks within the cluster.
  uint32_t group_origin_[3];
  uint32_t intra_offset_[3];
  // Cluster ownership: a cluster's members are reserved for whichever core took
  // its first CTA, so they cannot be distributed apart.
  bool     cluster_locked_;
  uint32_t cluster_core_;
  // CTAs sent but not yet admitted by each core's dispatcher.
  std::vector<uint32_t> inflight_;

  friend class SimObject<Kmu>;
};

} // namespace vortex
