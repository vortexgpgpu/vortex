#pragma once

#include <simobject.h>
#include "memsim.h"

namespace vortex {

struct CacheConfig {
    uint8_t C;              // log2 cache size    
    uint8_t B;              // log2 block size   
    uint8_t W;              // log2 word size 
    uint8_t A;              // log2 associativity    
    uint8_t addr_width;     // word address bits
    uint8_t num_banks;      // number of banks
    uint8_t ports_per_bank; // number of ports per bank
    uint8_t num_inputs;     // number of inputs
    bool    write_through;  // is write-through
    bool    write_reponse;  // enable write response
    uint16_t victim_size;   // victim cache size
    uint16_t mshr_size;     // MSHR buffer size
    uint8_t latency;        // pipeline latency 
};

class Cache : public SimObject<Cache> {  
public:
    Cache(const SimContext& ctx, const char* name, const CacheConfig& config);
    ~Cache();

    void step(uint64_t cycle);

    std::vector<SlavePort<MemReq>>  CoreReqPorts;
    std::vector<MasterPort<MemRsp>> CoreRspPorts;
    MasterPort<MemReq>              MemReqPort;
    SlavePort<MemRsp>               MemRspPort;
    
private:
    class Impl;
    Impl* impl_;
};

}