#pragma once

#include "types.h"

#ifndef MEM_LATENCY
#define MEM_LATENCY 24
#endif

#define RAM_PAGE_SIZE 4096

namespace vortex {

enum Constants {

    SMEM_BANK_OFFSET = log2ceil(sizeof(Word)) + log2ceil(STACK_SIZE / sizeof(Word)),

};

}