#include <vx_print.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

static const char* skip_flags(const char* format) {
	for (;;) {
		int c = *format++;
		switch (c) {
			case '-':
			case '+':
			case ' ':
			case '#': break;
			default : {
				return --format;
			}
		}
	}
	return NULL;	
}

static const char* skip_width(const char* format) {
	if (*format == '*') {
		++format;
	} else {
		char *endptr;
		strtol(format, &endptr, 10);
		format = endptr;
	}
	return format;
}

static const char* skip_precision(const char* format) {
	if (*format == '.') {
		++format;
		if (*format == '*') {
			++format;
		} else {
			char *endptr;
			strtol(format, &endptr, 10);
			format = endptr;
		}
	}
	return format;
}

static const char* skip_modifier(const char* format) {
	switch (*format) {
	case 'h':
		format++;
		if (*format == 'h') {
			format++;
		}
		break;
	case 'l':
		++format;
		if (*format == 'l') {
			++format;
		}
		break;
	case 'j':
	case 'z':
    case 't':
	case 'L':
		++format;
		break;
	default:
		break;
	}
	return format;
}

static const char* parse_format(const char* format, va_list va) {
	char buffer[64];
	char fmt[64];

	const char* p = format;
	p = skip_flags(p);	
	p = skip_width(p);
	p = skip_precision(p);
	p = skip_modifier(p);
	++p;

	int i;

	fmt[0] = '%';	
	for (i = 0; i < (p - format); ++i) {
		fmt[i+1] = format[i];
	}
	fmt[i+1] = 0;

	int len = vsnprintf(buffer, 256, fmt, va);

	for (i = 0; i < len; ++i) {
		vx_putchar(buffer[i]);
	}

	return p;
}

int vx_vprintf(const char* format, va_list va) {
	if (format == NULL)
		return -1;

	const char* p = format;
	int c = *p++;
	while (c) {		
  	if (c == '%') {			
			p = parse_format(p, va);
			c = *p++;	
		} else {
			vx_putchar(c);
			c = *p++;
		}		
	}

	return (int)(p - format);
}

int vx_printf(const char * format, ...) {
	va_list va;
  	va_start(va, format);
	int ret = vx_vprintf(format, va);
  	va_end(va);
  	return ret;
}

static const char hextoa[] = "0123456789abcdef";

void vx_prints(const char * str) {
	int c = *str++;
	while (c) {
		vx_putchar(c);
		c = *str++;
	}
}

void vx_printx(unsigned value) {
	if (value < 16) {
		vx_putchar(hextoa[value]);
	} else {
		int i = 32;
		bool start = false;
		do {
			int temp = (value >> (i - 4)) & 0xf;
			if (temp != 0) 
				start = true;
			if (start) 
				vx_putchar(hextoa[temp]);
			i-= 4;
		} while (i != 0);	
	}
	vx_putchar('\n');
}

void vx_printv(const char * str, unsigned value) {
	vx_prints(str);
	vx_printx(value);
}

#ifdef __cplusplus
}
#endif