#ifndef _TEXSW_H_

#include "common.h"

#define TEX_LOD_MAX 11

#define MIN(x, y)   ((x < y) ? (x) : (y))

#define MAX(x, y)   ((x > y) ? (x) : (y))

inline int address(int wrap, int value) {
    switch (wrap) {
    case 1: return value & 0xfffff;
    default:
    case 0: return MIN(MAX(value, 0), 0xfffff);
    }
}

inline void unpack(int format, int value, int* l, int* h) {
    switch (format) {
    case 1:
    case 2:
        *l = value;
        *h = 0;
        break;
    case 3:
        *l = (value | (value << 8)) & 0x00ff00ff;
        *h = 0;
        break;
    case 4:
        *l = (value | (value << 16)) & 0x07e0f81f;
        *h = 0;
        break;
    case 5:
        *l = (value | (value << 12)) & 0x0f0f0f0f;
        *h = 0;
        break;
    default:
    case 0: 
        *l = value & 0x00ff00ff;
        *h = (value >> 8) & 0x00ff00ff;
        break;
    }
}

inline void lerp(int al, int ah, int bl, int bh, int frac, int* l, int* h) {
    *l = (al + (((bl - al) * frac) >> 8)) & 0x00ff00ff;
    *h = (ah + (((bh - ah) * frac) >> 8)) & 0x00ff00ff;
}

inline int pack(int format, int l, int h) {
    switch (format) {
    case 1:
    case 2:
        return l;
    case 3:
        return (l | (l >> 8)) & 0xffff;
    case 4:
         return (l | (l >> 16)) & 0xffff;
    case 5:
        return (l | (l >> 12)) & 0xffff;
    default:
    case 0: 
        return (h << 8) | l;
    }
}

inline int tex_sw(kernel_arg_t* state, int stage, int u, int v, int lod) {
    int base_addr  = state->src_ptr;
	int mip_offset = 0;
	int log_width  = state->src_logWidth;
	int log_height = state->src_logHeight;
	int format     = state->format;
	int wrap       = state->wrap;
	int filter     = state->filter;

    int32_t* pBits = ((uint32_t*)base_addr) + mip_offset;    

    if (filter) {
        int u0 = address(wrap, u - (0x80000 >> log_width));
        int v0 = address(wrap, v - (0x80000 >> log_height)); 
        int u1 = address(wrap, u + (0x80000 >> log_width));    
        int v1 = address(wrap, v + (0x80000 >> log_height));

        int x0 = u0 >> (20 - log_width);
        int y0 = v0 >> (20 - log_height);
        int x1 = u1 >> (20 - log_width);
        int y1 = v1 >> (20 - log_height); 

        // memory lookup

        int c0 = pBits[x0 + (y0 << log_width)];
        int c1 = pBits[x1 + (y0 << log_width)];
        int c2 = pBits[x0 + (y1 << log_width)];
        int c3 = pBits[x1 + (y1 << log_width)];

        // filtering

        int alpha = x0 & 0xff;
        int beta  = y0 & 0xff;

        int c0a, c0b;  
        int c1a, c1b;
        int c01a, c01b;

        unpack(format, c0, &c0a, &c0b);
        unpack(format, c1, &c1a, &c1b);
        lerp(c0a, c0b, c1a, c1b, alpha, &c01a, &c01b);

        int c2a, c2b;  
        int c3a, c3b;
        int c23a, c23b;

        unpack(format, c2, &c2a, &c2b);
        unpack(format, c3, &c3a, &c3b);
        lerp(c2a, c2b, c3a, c3b, alpha, &c23a, &c23b);

        int c4a, c4b;
        lerp(c01a, c01b, c23a, c23b, beta, &c4a, &c4b);
        return pack(format, c4a, c4b);
    } else {
        int u0 = address(wrap, u);
        int v0 = address(wrap, v);  

        int x0 = u0 >> (20 - log_width);
        int y0 = v0 >> (20 - log_height);  

        int c0 = pBits[x0 + (y0 <<log_width)];

        int c0a, c0b;  
        unpack(format, c0, &c0a, &c0b);
        return pack(format, c0a, c0b);
    }
}

inline int vx_tex3(int stage, int u, int v, int lod) {
    int lodn = MIN(lod + 0x100000, TEX_LOD_MAX);
    int a = vx_tex(0, u, v, lod);  
    int b = vx_tex(0, u, v, lodn);  
    int al = a & 0x00ff00ff;
    int ah = (a >> 8) & 0x00ff00ff;    
    int bl = b & 0x00ff00ff;
    int bh = (b >> 8) & 0x00ff00ff;
    int frac = (lod >> 12) & 0xff;
    int cl = (al + (((bl - al) * frac) >> 8)) & 0x00ff00ff;
    int ch = (ah + (((bh - ah) * frac) >> 8)) & 0x00ff00ff;
    int c = al | (ah << 8);
    return c;
}

inline int tex3_sw(kernel_arg_t* state, int stage, int u, int v, int lod) {
    int lodn = MIN(lod + 0x10000, TEX_LOD_MAX);
    int a = tex_sw(state, 0, u, v, lod);    
    int b = tex_sw(state, 0, u, v, lodn);
    int al = a & 0x00ff00ff;
    int ah = (a >> 8) & 0x00ff00ff;
    
    int bl = b & 0x00ff00ff;
    int bh = (b >> 8) & 0x00ff00ff;
    int frac = (lod >> 12) & 0xff;
    int cl = (al + (((bl - al) * frac) >> 8)) & 0x00ff00ff;
    int ch = (ah + (((bh - ah) * frac) >> 8)) & 0x00ff00ff;
    int c = al | (ah << 8);
    return c;
}

#endif