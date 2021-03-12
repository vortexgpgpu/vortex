#ifndef VX_API_H
#define VX_API_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int vx_tex(unsigned t, unsigned u, unsigned v, unsigned lod);

#ifdef __cplusplus
}
#endif

#endif