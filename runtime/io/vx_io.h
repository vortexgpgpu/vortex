
#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

static char * hextoa[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"};
void vx_print_hex(unsigned);
void vx_printf(const char *, unsigned);

void vx_print_str(const char *);
void vx_printc(unsigned, char c);


#ifdef __cplusplus
}
#endif