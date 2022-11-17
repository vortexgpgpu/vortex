#ifndef __VX_PRINT_H__
#define __VX_PRINT_H__

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

int vx_vprintf(const char* format, va_list va);
int vx_printf(const char * format, ...);

void vx_putchar(int c);
void vx_putint(int value, int base);
void vx_putfloat(float value, int precision);

#ifdef __cplusplus
}
#endif

#endif // __VX_PRINT_H__
