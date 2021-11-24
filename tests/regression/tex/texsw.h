#pragma once

#include <vx_intrinsics.h>
#include <texturing.h>
#include "common.h"

inline uint32_t texel_read(uint8_t* address, uint32_t stride) {
    switch (stride) {
    case 1: return *(uint8_t*)address;
    case 2: return *(uint16_t*)address;
    case 4: return *(uint32_t*)address;
    default: 
        std::abort();
        return 0;
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
        TexAddressLinear(xu, xv, log_width, log_height, wrapu, wrapv, 
            &offset00, &offset01, &offset10, &offset11, &alpha, &beta);

        uint8_t* addr00 = base_addr + offset00 * stride;
        uint8_t* addr01 = base_addr + offset01 * stride;
        uint8_t* addr10 = base_addr + offset10 * stride;
        uint8_t* addr11 = base_addr + offset11 * stride;

        // memory lookup
        uint32_t texel00 = texel_read(addr00, stride);
        uint32_t texel01 = texel_read(addr01, stride);
        uint32_t texel10 = texel_read(addr10, stride);
        uint32_t texel11 = texel_read(addr11, stride);

        // filtering
        color = TexFilterLinear(
            format, texel00, texel01, texel10, texel11, alpha, beta);
    } else {
        // addressing
        uint32_t offset;
        TexAddressPoint(xu, xv, log_width, log_height, wrapu, wrapv, &offset);
        
        uint8_t* addr = base_addr + offset * stride;
        
        // memory lookup
        uint32_t texel = texel_read(addr, stride);

        // filtering
        color = TexFilterPoint(format, texel);
    }
    return color;
}

inline uint32_t tex_load_hw(kernel_arg_t* state, 
                            Fixed<TEX_FXD_FRAC> xu, 
                            Fixed<TEX_FXD_FRAC> xv, 
                            Fixed<16> xlod) {
    uint32_t color;
    int32_t ilod = std::max<int32_t>(xlod.data(), Fixed<16>::ONE);
    uint32_t lod = std::min<uint32_t>(log2floor(ilod) - 16, TEX_LOD_MAX);
    if (state->filter == 2) {        
        uint32_t lod_n  = std::min<uint32_t>(lod + 1, TEX_LOD_MAX);
        uint32_t frac   = ilod >> (lod + 16 - 8);
        uint32_t texel0 = vx_tex(0, xu.data(), xv.data(), lod); 
        uint32_t texel1 = vx_tex(0, xu.data(), xv.data(), lod_n);
        uint32_t cl, ch;
        {
            uint32_t c0l, c0h;  
            uint32_t c1l, c1h;
            Unpack8888(TexFormat::R8G8B8A8, texel0, &c0l, &c0h);
            Unpack8888(TexFormat::R8G8B8A8, texel1, &c1l, &c1h);
            Lerp8888(c0l, c0h, c1l, c1h, frac, &cl, &ch);
        }
        color = Pack8888(TexFormat::R8G8B8A8, cl, ch);
    } else {
        color = vx_tex(0, xu.data(), xv.data(), lod);
    }
    return color;
}

inline uint32_t tex_load_sw(kernel_arg_t* state, 
                            Fixed<TEX_FXD_FRAC> xu, 
                            Fixed<TEX_FXD_FRAC> xv, 
                            Fixed<16> xlod) {
    uint32_t color;
    int32_t ilod = std::max<int32_t>(xlod.data(), Fixed<16>::ONE);
    uint32_t lod = std::min<uint32_t>(log2floor(ilod) - 16, TEX_LOD_MAX);
    if (state->filter == 2) {        
        uint32_t lod_n  = std::min<uint32_t>(lod + 1, TEX_LOD_MAX);
        uint32_t frac   = ilod >> (lod + 16 - 8);
        uint32_t texel0 = vx_tex_sw(state, xu, xv, lod);  
        uint32_t texel1 = vx_tex_sw(state, xu, xv, lod_n);
        uint32_t cl, ch;
        {
            uint32_t c0l, c0h;  
            uint32_t c1l, c1h;
            Unpack8888(TexFormat::R8G8B8A8, texel0, &c0l, &c0h);
            Unpack8888(TexFormat::R8G8B8A8, texel1, &c1l, &c1h);
            Lerp8888(c0l, c0h, c1l, c1h, frac, &cl, &ch);
        }
        color = Pack8888(TexFormat::R8G8B8A8, cl, ch);
    } else {
        color = vx_tex_sw(state, xu, xv, lod);
    }
    return color;
}