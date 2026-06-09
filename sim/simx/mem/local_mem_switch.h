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

class LocalMemSwitch : public SimObject<LocalMemSwitch> {
public:
  using Ptr = std::shared_ptr<LocalMemSwitch>;

  SimChannel<LsuReq> ReqIn;
  SimChannel<LsuRsp> RspOut;

  SimChannel<LsuReq> ReqOutLmem;
  SimChannel<LsuRsp> RspInLmem;

  SimChannel<LsuReq> ReqOutDC;
  SimChannel<LsuRsp> RspInDC;

  LocalMemSwitch(
    const SimContext& ctx,
    const char* name,
    uint32_t delay
  );

protected:
  void on_tick();

private:
  uint32_t delay_;

  friend class SimObject<LocalMemSwitch>;
};

}
