#ifndef VX_IO_H
#define VX_IO_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void vx_print_hex(unsigned);
void vx_printf(const char *, unsigned);

void vx_print_str(const char *);
void vx_printc(unsigned, char c);

#ifdef __cplusplus
}
#endif

#endif