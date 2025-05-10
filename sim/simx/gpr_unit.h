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

#include "instr_trace.h"

namespace vortex {

struct GprReq {
  uint32_t opd;
  uint32_t rid;
  uint32_t wid;

  friend std::ostream& operator<<(std::ostream& os, const GprReq& req) {
    os << "opd=" << req.opd << ", rid=" << req.rid << ", wid=" << req.wid;
    return os;
  }
};

struct GprRsp {
  uint32_t opd;

  friend std::ostream& operator<<(std::ostream& os, const GprRsp& rsp) {
    os << "opd=" << rsp.opd;
    return os;
  }
};

template <uint32_t NUM_REQS, uint32_t NUM_BANKS>
class GprUnit : public SimObject<GprUnit<NUM_REQS, NUM_BANKS>> {
public:

  using ReqXbar = TxCrossBar<GprReq>;

  std::array<SimPort<GprReq>, NUM_REQS> ReqIn;
  std::array<SimPort<GprRsp>, NUM_REQS> RspOut;

  GprUnit(const SimContext &ctx)
    : SimObject<GprUnit<NUM_REQS, NUM_BANKS>>(ctx, "gpr-unit")
    , ReqIn(make_array<SimPort<GprReq>, NUM_REQS>(this))
    , RspOut(make_array<SimPort<GprRsp>, NUM_REQS>(this)) {
    char sname[100];
		snprintf(sname, 100, "%s-xbar", this->name().c_str());
    crossbar_ = ReqXbar::Create(sname, NUM_REQS, NUM_BANKS,
		 [](const GprReq& req)->uint32_t {
      //uint32_t bank_id = ((req.wid & BANKID_WIS_MASK) << BANKID_REG_BITS) | (req.rid & BANKID_REG_MASK);
			return req.rid % NUM_BANKS;
		});
    for (uint32_t i = 0; i < NUM_REQS; ++i) {
			ReqIn.at(i).bind(&crossbar_->Inputs.at(i));
		}
  }

  virtual ~GprUnit() {}

  virtual void reset() {
    //--
  }

  virtual void tick() {
    if (ReqIn.empty())
      return;
    for (uint32_t b = 0; b < NUM_BANKS; b++) {
      auto& output = crossbar_->Outputs.at(b);
      if (output.empty())
        continue;
      auto& req = output.front();
      GprRsp rsp;
      rsp.opd = req.data.opd;
      RspOut.at(req.input).push(rsp);
      output.pop();
    }
  }

  uint32_t total_stalls() const {
    return crossbar_->collisions();
  }

private:
  typename ReqXbar::Ptr crossbar_;
  constexpr static uint32_t BANK_SEL_BITS    = log2ceil(NUM_BANKS);
  constexpr static uint32_t _BANKID_WIS_BITS = std::min<uint32_t>(ISSUE_WIS_BITS, BANK_SEL_BITS - (BANK_SEL_BITS / 2));
  constexpr static uint32_t BANKID_WIS_BITS  = (BANK_SEL_BITS > 1 && ISSUE_WIS_BITS != 0) ? _BANKID_WIS_BITS : 0;
  constexpr static uint32_t BANKID_REG_BITS  = BANK_SEL_BITS - BANKID_WIS_BITS;
  constexpr static uint32_t BANKID_REG_MASK  = (1 << BANKID_REG_BITS) - 1;
  constexpr static uint32_t BANKID_WIS_MASK  = (1 << BANKID_WIS_BITS) - 1;
};

#ifdef EXT_V_ENABLE
typedef GprUnit<(NUM_OPCS + NUM_VOPCS), NUM_GPR_BANKS> GPR;
typedef GprUnit<NUM_OPCS, NUM_VGPR_BANKS> VGPR;
#else
typedef GprUnit<NUM_OPCS, NUM_GPR_BANKS> GPR;
#endif

} // namespace vortex
