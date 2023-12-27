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

class Scoreboard {
public:

    struct reg_use_t {
        RegType  reg_type;
        uint32_t reg_id; 
        ExeType  exe_type;
        SfuType  sfu_type;        
        uint64_t uuid;
    };
        
    Scoreboard(const Arch &arch) 
        : in_use_iregs_(arch.num_warps())
        , in_use_fregs_(arch.num_warps())
        , in_use_vregs_(arch.num_warps())
    {
        this->clear();
    }

    void clear() {
        for (uint32_t i = 0, n = in_use_iregs_.size(); i < n; ++i) {
            in_use_iregs_.at(i).reset();
            in_use_fregs_.at(i).reset();
            in_use_vregs_.at(i).reset();
        }
        owners_.clear();
    }

    bool in_use(pipeline_trace_t* trace) const {
        return (trace->used_iregs & in_use_iregs_.at(trace->wid)) != 0 
            || (trace->used_fregs & in_use_fregs_.at(trace->wid)) != 0
            || (trace->used_vregs & in_use_vregs_.at(trace->wid)) != 0;
    }

    std::vector<reg_use_t> get_uses(pipeline_trace_t* trace) const {
        std::vector<reg_use_t> out;  
        
        auto used_iregs = trace->used_iregs & in_use_iregs_.at(trace->wid);
        auto used_fregs = trace->used_fregs & in_use_fregs_.at(trace->wid);
        auto used_vregs = trace->used_vregs & in_use_vregs_.at(trace->wid);

        for (uint32_t r = 0; r < MAX_NUM_REGS; ++r) {
            if (used_iregs.test(r)) {
                uint32_t tag = (r << 16) | (trace->wid << 4) | (int)RegType::Integer;
                auto owner = owners_.at(tag);
                out.push_back({RegType::Integer, r, owner->exe_type, owner->sfu_type, owner->uuid});
            }
        }

        for (uint32_t r = 0; r < MAX_NUM_REGS; ++r) {
            if (used_fregs.test(r)) {
                uint32_t tag = (r << 16) | (trace->wid << 4) | (int)RegType::Float;
                auto owner = owners_.at(tag);
                out.push_back({RegType::Float, r, owner->exe_type, owner->sfu_type, owner->uuid});
            }
        }

        for (uint32_t r = 0; r < MAX_NUM_REGS; ++r) {
            if (used_vregs.test(r)) {
                uint32_t tag = (r << 16) | (trace->wid << 4) | (int)RegType::Vector;
                auto owner = owners_.at(tag);
                out.push_back({RegType::Vector, r, owner->exe_type, owner->sfu_type, owner->uuid});
            }
        }

        return out;
    }
    
    void reserve(pipeline_trace_t* trace) {
        assert(trace->wb);  
        switch (trace->rdest_type) {
        case RegType::Integer:            
            in_use_iregs_.at(trace->wid).set(trace->rdest);
            break;
        case RegType::Float:
            in_use_fregs_.at(trace->wid).set(trace->rdest);
            break;
        case RegType::Vector:
            in_use_vregs_.at(trace->wid).set(trace->rdest);
            break;
        default: assert(false);
        }      
        uint32_t tag = (trace->rdest << 16) | (trace->wid << 4) | (int)trace->rdest_type;
        assert(owners_.count(tag) == 0);
        owners_[tag] = trace;
        assert((int)trace->exe_type < 5);
    }

    void release(pipeline_trace_t* trace) {
        assert(trace->wb);      
        switch (trace->rdest_type) {
        case RegType::Integer:
            in_use_iregs_.at(trace->wid).reset(trace->rdest);
            break;
        case RegType::Float:
            in_use_fregs_.at(trace->wid).reset(trace->rdest);
            break;
        case RegType::Vector:
            in_use_vregs_.at(trace->wid).reset(trace->rdest);
            break;
        default: assert(false);
        }      
        uint32_t tag = (trace->rdest << 16) | (trace->wid << 4) | (int)trace->rdest_type;
        owners_.erase(tag);
    }

private:

    std::vector<RegMask> in_use_iregs_;
    std::vector<RegMask> in_use_fregs_;
    std::vector<RegMask> in_use_vregs_;
    std::unordered_map<uint32_t, pipeline_trace_t*> owners_;
};

}