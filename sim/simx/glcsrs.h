#pragma once

#include <util.h>
#include "texunit.h"
#include "rasterunit.h"
#include "ropunit.h"

namespace vortex {

class GlobalCSRS {
public:
    GlobalCSRS();
    ~GlobalCSRS();

    void clear();

    void write(uint32_t addr, uint64_t value);

    TexUnit::CSRS tex_csrs;
    RasterUnit::CSRS raster_csrs;
    RopUnit::CSRS rop_csrs;
};

}