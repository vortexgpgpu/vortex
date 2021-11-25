#pragma once

#include <vx_intrinsics.h>
#include <texturing.h>
#include "common.h"

using namespace cocogfx;

inline void  texel_read(uint32_t* texels,
                        uint8_t** addresses,
                        uint32_t count,
                        uint32_t stride) {
    switch (stride) {
    case 1: 
        for (uint32_t i = 0; i < count; ++i) {
            texels[i] = *(uint8_t*)addresses[i];
        }
        break;
    case 2: 
        for (uint32_t i = 0; i < count; ++i) {
            texels[i] = *(uint16_t*)addresses[i];
        }
        break;
    case 4: 
        for (uint32_t i = 0; i < count; ++i) {
            texels[i] = *(uint32_t*)addresses[i];
        }
        break;
    default: 
        std::abort();
    }
}

inline uint32_t vx_tex_sw(kernel_arg_t* state, 
                          Fixed<TEX_FXD_FRAC> xu, 
                          Fixed<TEX_FXD_FRAC> xv, 
                          uint32_t lod) {
    uint8_t* base_addr  = ((uint8_t*)state->src_addr) + state->mip_offs[lod];
	uint32_t log_width  = std::max<int32_t>(state->src_logwidth - lod, 0);
	uint32_t log_height = std::max<int32_t>(state->src_logheight - lod, 0);
	auto format = (TexFormat)state->format;
	auto wrapu  = (WrapMode)state->wrapu;
    auto wrapv  = (WrapMode)state->wrapv;
	auto filter = state->filter;
    auto stride = Stride(format);    

    uint32_t color;

    if (filter) {
        // addressing
        uint32_t offset00, offset01, offset10, offset11;
        uint32_t alpha, beta;
        uint8_t* addr[4];
        uint32_t texel[4];

        TexAddressLinear(xu, xv, log_width, log_height, wrapu, wrapv, 
            &offset00, &offset01, &offset10, &offset11, &alpha, &beta);

        addr[0] = base_addr + offset00 * stride;
        addr[1] = base_addr + offset01 * stride;
        addr[2] = base_addr + offset10 * stride;
        addr[3] = base_addr + offset11 * stride;

        // memory fetch
        texel_read(texel, addr, 4, stride);

        // filtering
        color = TexFilterLinear(
            format, texel[0], texel[1], texel[2], texel[3], alpha, beta);
    } else {
        // addressing
        uint32_t offset;
        uint8_t* addr;
        uint32_t texel;

        TexAddressPoint(xu, xv, log_width, log_height, wrapu, wrapv, &offset);
        
        addr = base_addr + offset * stride;
        
        // memory fetch
        texel_read(&texel, &addr, 1, stride);

        // filtering
        color = TexFilterPoint(format, texel);
    }
    return color;
}

inline uint32_t tex_load(kernel_arg_t* state, 
                         Fixed<TEX_FXD_FRAC> xu,
                         Fixed<TEX_FXD_FRAC> xv,
                         Fixed<16> xj) {
    uint32_t color;
    uint32_t j = std::max<int32_t>(xj.data(), Fixed<16>::ONE);
    uint32_t l = std::min<uint32_t>(log2floor(j) - 16, TEX_LOD_MAX);
    if (state->filter == 2) {        
        uint32_t ln = std::min<uint32_t>(l + 1, TEX_LOD_MAX);
        uint32_t f  = (j - (1 << (l + 16))) >> (l + 16 - 8);
        uint32_t texel0, texel1;
        if (state->use_sw) {
            texel0 = vx_tex_sw(state, xu, xv, l);  
            texel1 = vx_tex_sw(state, xu, xv, ln);
        } else {
            texel0 = vx_tex(0, xu.data(), xv.data(), l);
            texel1 = vx_tex(0, xu.data(), xv.data(), ln);
        }
        uint32_t cl, ch;
        {
            uint32_t c0l, c0h, c1l, c1h;
            Unpack8888(texel0, &c0l, &c0h);
            Unpack8888(texel1, &c1l, &c1h);
            cl = Lerp8888(c0l, c1l, f);
            ch = Lerp8888(c0h, c1h, f);
        }
        color = Pack8888(cl, ch);
        //vx_printf("j=0x%x, l=%d, ln=%d, f=%d, texel0=0x%x, texel1=0x%x, color=0x%x\n", j, l, ln, f, texel0, texel1, color);
    } else {
        if (state->use_sw) {
            color = vx_tex_sw(state, xu, xv, l);
        } else {
            color = vx_tex(0, xu.data(), xv.data(), l);
        }
    }
    return color;
}