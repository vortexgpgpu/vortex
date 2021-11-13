#pragma once

#include "pipeline.h"
#include <queue>

namespace vortex {

class Scoreboard {
private:
    std::vector<RegMask> in_use_iregs_;
    std::vector<RegMask> in_use_fregs_;
    std::vector<RegMask> in_use_vregs_;

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
    
    void reserve(const pipeline_state_t& state) {
        if (!state.rdest)
            return;
        
        switch (state.rdest_type) {
        case 1:            
            in_use_iregs_.at(state.wid).set(state.rdest);
            break;
        case 2:
            in_use_fregs_.at(state.wid).set(state.rdest);
            break;
        case 3:
            in_use_vregs_.at(state.wid).set(state.rdest);
            break;
        default:  
            break;
        }
    }

    void release(const pipeline_state_t& state) {
        if (!state.rdest)
            return;
        switch (state.rdest_type) {
        case 1:
            in_use_iregs_.at(state.wid).reset(state.rdest);
            break;
        case 2:
            in_use_fregs_.at(state.wid).reset(state.rdest);
            break;
        case 3:
            in_use_vregs_.at(state.wid).reset(state.rdest);
            break;
        default:  
            break;
        }      
    }
};

}