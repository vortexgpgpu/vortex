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
        RegType  type;
        uint32_t reg;        
        uint64_t owner;
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

    bool in_use(pipeline_trace_t* state) const {
        return (state->used_iregs & in_use_iregs_.at(state->wid)) != 0 
            || (state->used_fregs & in_use_fregs_.at(state->wid)) != 0
            || (state->used_vregs & in_use_vregs_.at(state->wid)) != 0;
    }

    std::vector<reg_use_t> get_uses(pipeline_trace_t* state) const {
        std::vector<reg_use_t> out;        
        {
            uint32_t r = 0;
            auto used_iregs = state->used_iregs & in_use_iregs_.at(state->wid);        
            while (used_iregs.any()) {
                if (used_iregs.test(0)) {
                    uint32_t tag = (r << 16) | (state->wid << 4) | (int)RegType::Integer;
                    out.push_back({RegType::Integer, r, owners_.at(tag)});
                }
                used_iregs >>= 1;
                ++r;
            }
        }
        {
            uint32_t r = 0;
            auto used_fregs = state->used_fregs & in_use_fregs_.at(state->wid);
            while (used_fregs.any()) {
                if (used_fregs.test(0)) {
                    uint32_t tag = (r << 16) | (state->wid << 4) | (int)RegType::Float;
                    out.push_back({RegType::Float, r, owners_.at(tag)});
                }
                used_fregs >>= 1;
                ++r;
            }
        }
        {
            uint32_t r = 0;
            auto used_vregs = state->used_vregs & in_use_vregs_.at(state->wid);
            while (used_vregs.any()) {
                if (used_vregs.test(0)) {
                    uint32_t tag = (r << 16) | (state->wid << 4) | (int)RegType::Vector;
                    out.push_back({RegType::Vector, r, owners_.at(tag)});
                }
                used_vregs >>= 1;
                ++r;
            }
        }
        return out;
    }
    
    void reserve(pipeline_trace_t* state) {
        assert(state->wb);  
        switch (state->rdest_type) {
        case RegType::Integer:            
            in_use_iregs_.at(state->wid).set(state->rdest);
            break;
        case RegType::Float:
            in_use_fregs_.at(state->wid).set(state->rdest);
            break;
        case RegType::Vector:
            in_use_vregs_.at(state->wid).set(state->rdest);
            break;
        default:  
            break;
        }      
        uint32_t tag = (state->rdest << 16) | (state->wid << 4) | (int)state->rdest_type;
        assert(owners_.count(tag) == 0);
        owners_[tag] = state->uuid;
    }

    void release(pipeline_trace_t* state) {
        assert(state->wb);      
        switch (state->rdest_type) {
        case RegType::Integer:
            in_use_iregs_.at(state->wid).reset(state->rdest);
            break;
        case RegType::Float:
            in_use_fregs_.at(state->wid).reset(state->rdest);
            break;
        case RegType::Vector:
            in_use_vregs_.at(state->wid).reset(state->rdest);
            break;
        default:  
            break;
        }      
        uint32_t tag = (state->rdest << 16) | (state->wid << 4) | (int)state->rdest_type;
        owners_.erase(tag);
    }

private:

    std::vector<RegMask> in_use_iregs_;
    std::vector<RegMask> in_use_fregs_;
    std::vector<RegMask> in_use_vregs_;
    std::unordered_map<uint32_t, uint64_t> owners_;
};

}