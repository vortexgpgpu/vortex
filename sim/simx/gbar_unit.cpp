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

#include "gbar_unit.h"
#include <algorithm>
#include "debug.h"

using namespace vortex;

GbarUnit::GbarUnit(const SimContext& ctx, const char* name, uint32_t num_clusters)
  : SimObject(ctx, name)
  , arrive_in(this)
  , resume_out(num_clusters, this)
  , arrived_(VX_CFG_NUM_BARRIERS, 0)
  , accept_next_(0)
{
  arrive_in.bind(this, &GbarUnit::on_arrive);
}

void GbarUnit::on_reset() {
  std::fill(arrived_.begin(), arrived_.end(), 0);
  accept_next_ = 0;
}

// A core arrives once per generation, so a count of arrivals tracks the
// participants without a mask over every core of the device.
void GbarUnit::on_arrive(const GbarArrive& msg) {
  uint64_t now = SimPlatform::instance().cycles();
  uint64_t accept = std::max(now, accept_next_);
  accept_next_ = accept + 1;
  auto& arrived = arrived_.at(msg.bar_id % arrived_.size());
  ++arrived;
  DT(4, "*** Device barrier arrive: core #" << msg.core_id << " at barrier #" << msg.bar_id << ", arrived=" << arrived);
  if (arrived == msg.count) {
    for (auto& link : resume_out) {
      link.send({msg.bar_id}, accept - now + 1);
    }
    arrived = 0;
  }
}
