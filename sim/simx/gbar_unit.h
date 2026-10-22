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

#include <vector>
#include "types.h"

namespace vortex {

// Device-level global barrier: with more than one cluster, a barrier's
// participants span clusters, so each cluster forwards its cores' arrivals
// here and every cluster is resumed once the count is met. Arrivals enter
// through the device arbiter's output buffer, one per cycle; the release is
// registered in the unit and again in the response fork to the clusters.
class GbarUnit : public SimObject<GbarUnit> {
public:
  SimEventLink<GbarArrive> arrive_in;
  std::vector<SimEventLink<GbarResume>> resume_out;

  GbarUnit(const SimContext& ctx, const char* name, uint32_t num_clusters);

protected:
  void on_reset();

private:
  void on_arrive(const GbarArrive& msg);

  std::vector<uint32_t> arrived_;
  uint64_t accept_next_;  // first cycle the unit can take another arrival

  friend class SimObject<GbarUnit>;
};

} // namespace vortex
