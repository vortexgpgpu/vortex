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

namespace vortex {

class MemCoalescer : public SimObject<MemCoalescer> {
public:
  SimPort<LsuReq> ReqIn;
  SimPort<LsuRsp> RspIn;

  SimPort<LsuReq> ReqOut;
  SimPort<LsuRsp> RspOut;

  MemCoalescer(
    const SimContext& ctx,
    const char* name,
    uint32_t input_size,
    uint32_t output_size,
    uint32_t line_size,
    uint32_t queue_size,
    uint32_t delay
  );

  void reset();

  void tick();

private:

  struct pending_req_t {
    uint32_t tag;
    BitVector<> mask;
  };

  uint32_t input_size_;
  uint32_t output_size_;
  uint32_t output_ratio_;

  HashTable<pending_req_t> pending_rd_reqs_;
  BitVector<> sent_mask_;
  uint32_t line_size_;
  uint32_t delay_;
};

}