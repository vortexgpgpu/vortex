#include "rop_unit.h"
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

uint32_t RopUnit::get_state(uint32_t state) {
    return states_.at(state);
}

void RopUnit::set_state(uint32_t state, uint32_t value) {
    states_.at(state) = value;
}

void RopUnit::write(uint32_t x, uint32_t y, uint32_t color) {
    __unused (x);
    __unused (y);
    __unused (color);

    //--
}