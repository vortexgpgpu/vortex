#pragma once

#include "types.h"

namespace vortex {

class Core;

class TexUnit {
public:
    TexUnit(Core* core);
    ~TexUnit();

    void clear();

    uint32_t get_state(uint32_t state);
  
    void set_state(uint32_t state, uint32_t value);

    uint32_t read(int32_t u, int32_t v, int32_t lod, std::vector<mem_addr_size_t>* mem_addrs);

private:

    std::array<uint32_t, NUM_TEX_STATES> states_;
    Core* core_;
};

}