#pragma once

#include "pipeline.h"
#include <queue>

namespace vortex {

class Scoreboard {
private:
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
        for (int w = 0; w < arch.num_warps(); ++w) {    
            in_use_iregs_.at(w).reset();
            in_use_fregs_.at(w).reset();
            in_use_vregs_.at(w).reset();    
        }
    }

    bool in_use(const pipeline_state_t& state) const {
        return (state.used_iregs & in_use_iregs_.at(state.wid)) != 0 
            || (state.used_fregs & in_use_fregs_.at(state.wid)) != 0
            || (state.used_vregs & in_use_vregs_.at(state.wid)) != 0;
    }

    std::vector<uint64_t> owners(const pipeline_state_t& state) const {
        std::vector<uint64_t> out;        
        {
            uint32_t r = 0;
            auto used_iregs = state.used_iregs & in_use_iregs_.at(state.wid);        
            while (used_iregs.any()) {
                if (used_iregs.test(0)) {
                    uint32_t tag = (r << 16) | (state.wid << 4) | (int)RegType::Integer;
                    out.push_back(owners_.at(tag));
                }
                used_iregs >>= 1;
                ++r;
            }
        }
        {
            uint32_t r = 0;
            auto used_fregs = state.used_fregs & in_use_fregs_.at(state.wid);
            while (used_fregs.any()) {
                if (used_fregs.test(0)) {
                    uint32_t tag = (r << 16) | (state.wid << 4) | (int)RegType::Float;
                    out.push_back(owners_.at(tag));
                }
                used_fregs >>= 1;
                ++r;
            }
        }
        {
            uint32_t r = 0;
            auto used_vregs = state.used_vregs & in_use_vregs_.at(state.wid);
            while (used_vregs.any()) {
                if (used_vregs.test(0)) {
                    uint32_t tag = (r << 16) | (state.wid << 4) | (int)RegType::Vector;
                    out.push_back(owners_.at(tag));
                }
                used_vregs >>= 1;
                ++r;
            }
        }
        return std::move(out);
    }
    
    void reserve(const pipeline_state_t& state) {
        if (!state.wb)
            return;  
        switch (state.rdest_type) {
        case RegType::Integer:            
            in_use_iregs_.at(state.wid).set(state.rdest);
            break;
        case RegType::Float:
            in_use_fregs_.at(state.wid).set(state.rdest);
            break;
        case RegType::Vector:
            in_use_vregs_.at(state.wid).set(state.rdest);
            break;
        default:  
            break;
        }      
        uint32_t tag = (state.rdest << 16) | (state.wid << 4) | (int)state.rdest_type;
        assert(owners_.count(tag) == 0);
        owners_[tag] = state.id;
    }

    void release(const pipeline_state_t& state) {
        if (!state.wb)
            return;       
        switch (state.rdest_type) {
        case RegType::Integer:
            in_use_iregs_.at(state.wid).reset(state.rdest);
            break;
        case RegType::Float:
            in_use_fregs_.at(state.wid).reset(state.rdest);
            break;
        case RegType::Vector:
            in_use_vregs_.at(state.wid).reset(state.rdest);
            break;
        default:  
            break;
        }      
        uint32_t tag = (state.rdest << 16) | (state.wid << 4) | (int)state.rdest_type;
        owners_.erase(tag);
    }
};

}