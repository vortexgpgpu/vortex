#pragma once

#include <util.h>
#include "tex_unit.h"
#include "raster_unit.h"
#include "rop_unit.h"

namespace vortex {

class DCRS {
public:
    DCRS();
    ~DCRS();

    void clear();

    void write(uint32_t addr, uint64_t value);

    TexUnit::DCRS tex_dcrs;
    RasterUnit::DCRS raster_dcrs;
    RopUnit::DCRS rop_dcrs;
};

}