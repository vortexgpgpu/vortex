#pragma once

#include "types.h"

namespace vortex {

class Core;

class TexUnit {
public:
    TexUnit(Core* core);
    ~TexUnit();

    void clear();

    uint32_t csr_read(uint32_t addr);
  
    void csr_write(uint32_t addr, uint32_t value);

    uint32_t read(uint32_t stage, int32_t u, int32_t v, int32_t lod, std::vector<mem_addr_size_t>* mem_addrs);

private:

    std::array<std::array<uint32_t, NUM_TEX_STATES>, NUM_TEX_STAGES> states_;
    uint32_t csr_tex_unit_;
    Core* core_;
};

}