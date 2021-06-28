#ifndef VX_PRINT_H
#define VX_PRINT_H

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

int vx_vprintf(const char* format, va_list va);
int vx_printf(const char * format, ...);
int vx_putchar(int c);

#ifdef __cplusplus
}
#endif

#endif