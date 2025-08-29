// #ifdef __cplusplus
// extern "C" {
// #endif

#ifndef KERNEL_FIN_H
#define KERNEL_FIN_H

#include "common.h"

// Function declaration
void kernel_fin(fp *initvalu,
                int initvalu_offset_ecc,
                int initvalu_offset_Dyad,
                int initvalu_offset_SL,
                int initvalu_offset_Cyt,
                fp *parameter,
                fp *finavalu,
                fp JCaDyad,
                fp JCaSL,
                fp JCaCyt);

#endif // KERNEL_FIN_H

// #ifdef __cplusplus
// }
// #endif
