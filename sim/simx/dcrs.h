#pragma once

#include <util.h>
#include "texunit.h"
#include "rasterunit.h"
#include "ropunit.h"

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