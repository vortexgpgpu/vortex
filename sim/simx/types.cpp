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

#include "types.h"

using namespace vortex;

LocalMemDemux::LocalMemDemux(
  const SimContext& ctx, 
  const char* name, 
  uint32_t delay
) : SimObject<LocalMemDemux>(ctx, name)    
  , ReqIn(this)
  , RspIn(this)
  , ReqSM(this)
  , RspSM(this)
  , ReqDC(this)
  , RspDC(this)
  , delay_(delay)
{}

void LocalMemDemux::reset() {}

void LocalMemDemux::tick() {      
  // process incoming responses
  if (!RspSM.empty()) {
    auto& rsp = RspSM.front();
    DT(4, this->name() << "-" << rsp);
    RspIn.push(rsp, 1);
    RspSM.pop();
  }
  if (!RspDC.empty()) {
    auto& rsp = RspDC.front();
    DT(4, this->name() << "-" << rsp);
    RspIn.push(rsp, 1);
    RspDC
    .pop();
  }
  // process incoming requests  
  if (!ReqIn.empty()) {
    auto& req = ReqIn.front();
    DT(4, this->name() << "-" << req);
    if (req.type == AddrType::Shared) {
      ReqSM.push(req, delay_);
    } else {
      ReqDC.push(req, delay_);
    }
    ReqIn.pop();
  }   
}