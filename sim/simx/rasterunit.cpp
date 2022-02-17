#include "rasterunit.h"
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

uint32_t RasterUnit::csr_read(uint32_t addr) {
    uint32_t state = CSR_RASTER_STATE(addr);
    return states_.at(state);
}

void RasterUnit::csr_write(uint32_t addr, uint32_t value) {
    uint32_t state = CSR_RASTER_STATE(addr);
    states_.at(state) = value;
}

bool RasterUnit::pop(raster_quad_t* quad) {
    //--
    __unused (quad);

    return false;
}