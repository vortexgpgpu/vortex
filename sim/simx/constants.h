#pragma once

#include "types.h"

#define RAM_PAGE_SIZE 4096

#define DRAM_CHANNELS 2

namespace vortex {

enum Constants {

    SMEM_BANK_OFFSET = log2ceil(sizeof(Word)) + log2ceil(STACK_SIZE / sizeof(Word)),

};

}