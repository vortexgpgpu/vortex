#pragma once

#include "types.h"

namespace vortex {

class Core;

class RopUnit {
public:
    RopUnit(Core* core);
    ~RopUnit();

    void clear();

    uint32_t get_state(uint32_t state);
  
    void set_state(uint32_t state, uint32_t value);

    void write(uint32_t x, uint32_t y, uint32_t color);

private:

    std::array<uint32_t, NUM_ROP_STATES> states_;
    Core* core_;
};

}