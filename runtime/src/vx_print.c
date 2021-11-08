#include <vx_print.h>
#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	const char* format;
	va_list*    va;
	int         ret;
} printf_arg_t;

typedef struct {
	int value;
	int base;
} putint_arg_t;

typedef struct {
	float value;
	int precision;
} putfloat_arg_t;

static void __printf_cb(printf_arg_t* arg) {
	arg->ret = vprintf(arg->format, *arg->va);
}

int vx_vprintf(const char* format, va_list va) {
	printf_arg_t arg;
	arg.format = format;
	arg.va = &va;
	vx_serial((vx_serial_cb)__printf_cb, &arg);
  	return arg.ret;
}

int vx_printf(const char * format, ...) {
	int ret;
	va_list va;
	va_start(va, format);
	ret = vx_vprintf(format, va);
	va_end(va);		
  	return ret;
}

static void __putint_cb(const putint_arg_t* arg) {
	char tmp[33];
	float value = arg->value;
	int base = arg->base;
	itoa(value, tmp, base);
	for (int i = 0; i < 33; ++i) {
		int c = tmp[i];
		if (!c) break;
		vx_putchar(c);
	}
}

void vx_putint(int value, int base) {
	putint_arg_t arg;
	arg.value = value;
	arg.base = base;
	vx_serial((vx_serial_cb)__putint_cb, &arg);
}

static void __putfloat_cb(const putfloat_arg_t* arg) {
	float value = arg->value;
	int precision = arg->precision;
	int ipart = (int)value;
    vx_putint(ipart, 10);
    if (precision != 0) {
        vx_putchar('.');
		float frac = value - (float)ipart;
        float fscaled = frac * pow(10, precision);  
        vx_putint((int)fscaled, 10);
    }
}

void vx_putfloat(float value, int precision) {
	putfloat_arg_t arg;
	arg.value = value;
	arg.precision = precision;
	vx_serial((vx_serial_cb)__putfloat_cb, &arg);
}

#ifdef __cplusplus
}
#endif