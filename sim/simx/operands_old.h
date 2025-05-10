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

class Operands : public SimObject<Operands> {
private:
		static constexpr uint32_t NUM_BANKS = 4;
		uint32_t total_stalls_ = 0;

public:
    SimPort<instr_trace_t*> Input;
    SimPort<instr_trace_t*> Output;

    Operands(const SimContext& ctx, Core* /*core*/)
			: SimObject<Operands>(ctx, "operands")
			, Input(this)
			, Output(this)
    {
			total_stalls_ = 0;
		}

    virtual ~Operands() {}

    virtual void reset() {
			total_stalls_ = 0;
		}

    virtual void tick() {
			if (Input.empty())
				return;
			auto trace = Input.front();

			uint32_t stalls = 0;

			for (uint32_t i = 0; i < NUM_SRC_REGS; ++i) {
				uint32_t x_rid = trace->src_regs[i].id();
				if (x_rid == 0)
					continue; // skip x0 or empty
				for (uint32_t j = i + 1; j < NUM_SRC_REGS; ++j) {
					uint32_t y_rid = trace->src_regs[j].id();
					if (y_rid == 0)
						continue; // skip x0 or empty
					uint32_t bank_x = x_rid % NUM_BANKS;
					uint32_t bank_y = y_rid % NUM_BANKS;
					if (bank_x == bank_y) {
						++stalls;
					}
				}
			}

			total_stalls_ += stalls;

			Output.push(trace, 2 + stalls);

			DT(3, "pipeline-operands: " << *trace);

			Input.pop();
    };

		bool writeback(instr_trace_t* trace) {
			__unused(trace);
			return true;
		}

		uint32_t total_stalls() const {
			return total_stalls_;
		}
};

}