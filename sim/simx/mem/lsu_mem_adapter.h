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

#include "types.h"

namespace vortex {

class LsuMemAdapter : public SimObject<LsuMemAdapter> {
public:
  using Ptr = std::shared_ptr<LsuMemAdapter>;

  SimChannel<LsuReq> ReqIn;
  SimChannel<LsuRsp> RspOut;

  std::vector<SimChannel<MemReq>> ReqOut;
  std::vector<SimChannel<MemRsp>> RspIn;

  LsuMemAdapter(
    const SimContext& ctx,
    const char* name,
    uint32_t num_inputs,
    uint32_t delay
  );

  LsuMemAdapter(
    const SimContext& ctx,
    const char* name,
    uint32_t num_inputs
  ) : LsuMemAdapter(ctx, name, num_inputs, 0)
  {}

protected:
  void on_reset();
  void on_tick();

private:
  uint32_t delay_;
  BitVector<uint32_t> pending_mask_;

  friend class SimObject<LsuMemAdapter>;
};

}
