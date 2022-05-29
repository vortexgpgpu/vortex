#pragma once

#include <util.h>
#include "tex_unit.h"
#include "raster_unit.h"
#include "rop_unit.h"

namespace vortex {

class BaseDCRS {
private:    
    std::array<uint32_t, DCR_BASE_STATE_COUNT> states_;

public:
    BaseDCRS() {
        this->clear();
    }

    void clear() {
        for (auto& state : states_) {
            state = 0;
        }
        this->write(DCR_STARTUP_ADDR, STARTUP_ADDR);
    }

    uint32_t read(uint32_t addr) const {
        uint32_t state = DCR_BASE_STATE(addr);
        return states_.at(state);
    }

    void write(uint32_t addr, uint32_t value) {
        uint32_t state = DCR_BASE_STATE(addr);
        states_.at(state) = value;
    }
};

class DCRS {
public:
    DCRS();
    ~DCRS();

    void clear();

    void write(uint32_t addr, uint64_t value);
    
    BaseDCRS        base_dcrs;
    TexUnit::DCRS   tex_dcrs;
    RasterUnit::DCRS raster_dcrs;
    RopUnit::DCRS   rop_dcrs;
};

}