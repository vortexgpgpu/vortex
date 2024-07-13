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

class Operand : public SimObject<Operand> {
private:
		static constexpr uint32_t NUM_BANKS = 4;
		uint32_t total_stalls_ = 0;

public:
    SimPort<instr_trace_t*> Input;
    SimPort<instr_trace_t*> Output;

    Operand(const SimContext& ctx)
			: SimObject<Operand>(ctx, "Operand")
			, Input(this)
			, Output(this)
    {
			total_stalls_ = 0;
		}

    virtual ~Operand() {}

    virtual void reset() {
			total_stalls_ = 0;
		}

    virtual void tick() {
			if (Input.empty())
				return;
			auto trace = Input.front();

			uint32_t stalls = 0;

			for (int i = 0; i < NUM_SRC_REGS; ++i) {
				for (int j = i + 1; j < NUM_SRC_REGS; ++j) {
					int bank_i = trace->src_regs[i].idx % NUM_BANKS;
					int bank_j = trace->src_regs[j].idx % NUM_BANKS;
					if ((trace->src_regs[i].type != RegType::None)
					 && (trace->src_regs[j].type != RegType::None)
					 && (trace->src_regs[i].idx != 0)
					 && (trace->src_regs[j].idx != 0)
					 && bank_i == bank_j) {
						++stalls;
					}
				}
			}

			total_stalls_ += stalls;

			Output.push(trace, 2 + stalls);

			DT(3, "pipeline-operands: " << *trace);

			Input.pop();
    };

		uint32_t total_stalls() const {
			return total_stalls_;
		}
};

}