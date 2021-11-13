#pragma once

#include "types.h"

#ifndef MEM_LATENCY
#define MEM_LATENCY 18
#endif

namespace vortex {

struct Constants {

static constexpr uint32_t CORE_TO_DCACHE_DELAY = 1 + SM_ENABLE;
static constexpr uint32_t CORE_TO_ICACHE_DELAY = 1;

static constexpr uint32_t ICACHE_TO_MEM_DELAY = 2;
static constexpr uint32_t DCACHE_TO_MEM_DELAY = 2;

};

}