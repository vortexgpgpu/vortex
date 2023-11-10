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

#include <simobject.h>
#include "types.h"

namespace vortex {

class MemSim : public SimObject<MemSim>{
public:
    struct Config {        
        uint32_t channels;      
        uint32_t num_cores;
    };

    struct PerfStats {
        uint64_t reads;
        uint64_t writes;

        PerfStats() 
            : reads(0)
            , writes(0)
        {}
    };

    SimPort<MemReq> MemReqPort;
    SimPort<MemRsp> MemRspPort;

    MemSim(const SimContext& ctx, const char* name, const Config& config);
    ~MemSim();

    void reset();

    void tick();

    const PerfStats& perf_stats() const;
    
private:
    class Impl;
    Impl* impl_;
};

};