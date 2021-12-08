#pragma once

#ifndef RAM_PAGE_SIZE
#define RAM_PAGE_SIZE 4096
#endif

#ifndef MEM_CYCLE_RATIO
#define MEM_CYCLE_RATIO -1
#endif

#ifndef MEMORY_BANKS
#define MEMORY_BANKS 2
#endif

namespace vortex {

enum Constants {

    SMEM_BANK_OFFSET = log2ceil(sizeof(Word)) + log2ceil(STACK_SIZE / sizeof(Word)),

};

}