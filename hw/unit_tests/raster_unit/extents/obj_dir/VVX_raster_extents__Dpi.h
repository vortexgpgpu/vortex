// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Prototypes for DPI import and export functions.
//
// Verilator includes this file in all generated .cpp files that use DPI functions.
// Manually include this file where DPI .c import functions are declared to ensure
// the C functions match the expectations of the DPI imports.

#include "svdpi.h"

#ifdef __cplusplus
extern "C" {
#endif
    
    
    // DPI IMPORTS
    // DPI import at ../../../dpi/util_dpi.vh:8:30
    extern void dpi_assert(int inst, svLogic cond, int delay);
    // DPI import at ../../../dpi/util_dpi.vh:5:30
    extern void dpi_idiv(svLogic enable, int a, int b, svLogic is_signed, int* quotient, int* remainder);
    // DPI import at ../../../dpi/util_dpi.vh:4:30
    extern void dpi_imul(svLogic enable, int a, int b, svLogic is_signed_a, svLogic is_signed_b, int* resultl, int* resulth);
    // DPI import at ../../../dpi/util_dpi.vh:7:29
    extern int dpi_register();
    // DPI import at ../../../dpi/util_dpi.vh:10:30
    extern void dpi_trace(const char* format);
    // DPI import at ../../../dpi/util_dpi.vh:11:30
    extern void dpi_trace_start();
    // DPI import at ../../../dpi/util_dpi.vh:12:30
    extern void dpi_trace_stop();
    
#ifdef __cplusplus
}
#endif
