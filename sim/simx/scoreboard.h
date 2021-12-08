#pragma once

#include "pipeline.h"
#include <queue>

namespace vortex {

class Scoreboard {
private:
    struct reg_use_t {
        RegType  type;
        uint32_t reg;        
        uint64_t owner;
    };

    std::vector<RegMask> in_use_iregs_;
    std::vector<RegMask> in_use_fregs_;
    std::vector<RegMask> in_use_vregs_;
    std::unordered_map<uint32_t, uint64_t> owners_; 

public:    
    Scoreboard(const ArchDef &arch) 
        : in_use_iregs_(arch.num_warps())
        , in_use_fregs_(arch.num_warps())
        , in_use_vregs_(arch.num_warps())
    {
        this->clear();
    }

    void clear() {
        for (int i = 0, n = in_use_iregs_.size(); i < n; ++i) {
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
        return std::move(out);
    }
    
    void reserve(pipeline_trace_t* state) {
        if (!state->wb)
            return;  
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
        if (!state->wb)
            return;       
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
};

}