#include "ropunit.h"
#include "core.h"
#include <VX_config.h>

using namespace vortex;

RopUnit::RopUnit(Core* core) 
    : core_(core ) {
    //--
}

RopUnit::~RopUnit() {
    //--
}

void RopUnit::clear() {
    for (auto& state : states_) {
        state = 0;
    }
}

uint32_t RopUnit::csr_read(uint32_t addr) {
    uint32_t state = CSR_ROP_STATE(addr);
    return states_.at(state);
}

void RopUnit::csr_write(uint32_t addr, uint32_t value) {
    uint32_t state = CSR_ROP_STATE(addr);
    states_.at(state) = value;
}

void RopUnit::write(uint32_t x, uint32_t y, uint32_t color) {
    __unused (x);
    __unused (y);
    __unused (color);

    //--
}