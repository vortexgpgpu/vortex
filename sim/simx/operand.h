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

#include "pipeline.h"
#include <queue>

namespace vortex {

class Operand : public SimObject<Operand> {
public:
    SimPort<pipeline_trace_t*> Input;
    SimPort<pipeline_trace_t*> Output;

    Operand(const SimContext& ctx) 
        : SimObject<Operand>(ctx, "Operand") 
        , Input(this)
        , Output(this)
    {}
    
    virtual ~Operand() {}

    virtual void reset() {}

    virtual void tick() {
        if (Input.empty())
            return;
        auto trace = Input.front();

        int delay = 1;
        for (int i = 0; i < MAX_NUM_REGS; ++i) {
            bool is_iregs = trace->used_iregs.test(i);
            bool is_fregs = trace->used_fregs.test(i);
            bool is_vregs = trace->used_vregs.test(i);
            if (is_iregs || is_fregs || is_vregs) {
                if (is_iregs && i == 0)
                    continue;
                ++delay;
            }
        }

        Output.send(trace, delay);
        
        DT(3, "pipeline-operands: " << *trace);

        Input.pop();
    };
};

}