#pragma once

#include <simobject.h>
#include <vector>
#include <list>

namespace vortex {

struct MemReq {
    uint64_t addr;
    uint32_t tag;
    bool write;
    bool is_io;

    MemReq(uint64_t _addr = 0, 
           uint64_t _tag = 0, 
           bool _write = false, 
           bool _is_io = false
    )   : addr(_addr)
        , tag(_tag)
        , write(_write)
        , is_io(_is_io) 
    {}
};

struct MemRsp {
    uint64_t tag;    
    MemRsp(uint64_t _tag = 0) : tag (_tag) {}
};

class MemSim : public SimObject<MemSim>{
private:
    class Impl;
    Impl* impl_;

public:

    MemSim(const SimContext& ctx, uint32_t num_inputs, uint32_t latency);
    ~MemSim();

    void step(uint64_t cycle);

    std::vector<SlavePort<MemReq>>  MemReqPorts;
    std::vector<MasterPort<MemRsp>> MemRspPorts;
};

};