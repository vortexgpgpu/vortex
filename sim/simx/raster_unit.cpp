#include "raster_unit.h"
#include "core.h"
#include <VX_config.h>

using namespace vortex;

RasterUnit::RasterUnit(Core* core) 
    : core_(core ) {
    //--
}

RasterUnit::~RasterUnit() {
    //--
}

void RasterUnit::clear() {
    for (auto& state : states_) {
        state = 0;
    }
}

uint32_t RasterUnit::get_state(uint32_t state) {
    return states_.at(state);
}

void RasterUnit::set_state(uint32_t state, uint32_t value) {
    states_.at(state) = value;
}

bool RasterUnit::pop(raster_quad_t* quad) {
    //--
    __unused (quad);

    return false;
}