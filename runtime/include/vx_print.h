#ifndef VX_PRINT_H
#define VX_PRINT_H

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

void vx_prints(const char * str);
void vx_printx(unsigned value);
void vx_printv(const char * str, unsigned value);

int vx_vprintf(const char* format, va_list va);
int vx_printf(const char * format, ...);
int vx_putchar(int c);

#ifdef __cplusplus
}
#endif

#endif