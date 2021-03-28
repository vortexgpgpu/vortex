#ifndef VX_TEX_H
#define VX_TEX_H

#ifdef __cplusplus
extern "C" {
#endif

unsigned vx_tex(unsigned unit, unsigned u, unsigned v, unsigned lod) {
    unsigned result;
    unsigned lod_unit = (unit << 24) | lod;
    asm volatile (".insn r4 0x6b, 5, 0, %0, %1, %2, %3" : "=r"(result) : "r"(u), "r"(v), "r"(lod_unit));
    return result;
}

#ifdef __cplusplus
}
#endif

#endif