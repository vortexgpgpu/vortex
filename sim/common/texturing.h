#pragma once

#include <cstdint>
#include <cstdlib>
#include <fixed.h>
#include <bitmanip.h>

enum class WrapMode {
  Clamp,
  Repeat,
  Mirror,
};

enum class TexFormat {
  R8G8B8A8,
  R5G6B5,  
  R4G4B4A4,
  L8A8,
  L8,  
  A8,  
};

template <uint32_t F, typename T = int32_t>
T Clamp(Fixed<F,T> fx, WrapMode mode) {
  switch (mode) {
  case WrapMode::Clamp:  return (fx.data() < 0) ? 0 : ((fx.data() > Fixed<F,T>::MASK) ? Fixed<F,T>::MASK : fx.data());
  case WrapMode::Repeat: return (fx.data() & Fixed<F,T>::MASK);
  case WrapMode::Mirror: return (bit_get(fx.data(), Fixed<F,T>::FRAC) ? ~fx.data() : fx.data());
  default: 
    std::abort();
    return 0;    
  }
}

inline uint32_t Stride(TexFormat format) {
  switch (format) {
  case TexFormat::R8G8B8A8: 
    return 4;
  case TexFormat::R5G6B5:
  case TexFormat::R4G4B4A4:
  case TexFormat::L8A8:
    return 2;
  case TexFormat::L8:
  case TexFormat::A8:
    return 1;
  default: 
    std::abort();
    return 0;
  }
}

inline void Unpack8888(TexFormat format, 
                       uint32_t texel, 
                       uint32_t* lo, 
                       uint32_t* hi) {
  switch (format) {
  case TexFormat::R8G8B8A8: 
    *lo = texel & 0x00ff00ff;
    *hi = (texel >> 8) & 0x00ff00ff;
    break;
  case TexFormat::R5G6B5:
  case TexFormat::R4G4B4A4:
    *lo = texel;
    *hi= 0;
    break;
  case TexFormat::L8A8:
    *lo = (texel | (texel << 8)) & 0x00ff00ff;
    *hi = 0;
    break;
  case TexFormat::L8:
    *lo = (texel | (texel << 16)) & 0x07e0f81f;
    *hi = 0;
    break;
  case TexFormat::A8:
    *lo = (texel | (texel << 12)) & 0x0f0f0f0f;
    *hi = 0;
    break;
  default: 
    std::abort();
  }
}

inline uint32_t Pack8888(TexFormat format, uint32_t lo, uint32_t hi) {
  switch (format) {
  case TexFormat::R8G8B8A8: 
    return (hi << 8) | lo;
  case TexFormat::R5G6B5:
  case TexFormat::R4G4B4A4:
    return lo;
  case TexFormat::L8A8:
    return (lo | (lo >> 8)) & 0xffff;
  case TexFormat::L8:
    return (lo | (lo >> 16)) & 0xffff;
  case TexFormat::A8:
    return (lo | (lo >> 12)) & 0xffff;  
  default: 
    std::abort();
    return 0;
  }
}

inline void Lerp8888(uint32_t al, 
                     uint32_t ah, 
                     uint32_t bl, 
                     uint32_t bh, 
                     uint32_t frac, 
                     uint32_t* lo, 
                     uint32_t* hi) {
    *lo = (al + (((bl - al) * frac) >> 8)) & 0x00ff00ff;
    *hi = (ah + (((bh - ah) * frac) >> 8)) & 0x00ff00ff;
}

template <uint32_t F, typename T = int32_t>
void TexAddressLinear(Fixed<F,T> fu, 
                      Fixed<F,T> fv, 
                      uint32_t log_width,
                      uint32_t log_height,
                      WrapMode wrapu,
                      WrapMode wrapv,
                      uint32_t* addr00,
                      uint32_t* addr01,
                      uint32_t* addr10,
                      uint32_t* addr11,
                      uint32_t* alpha,
                      uint32_t* beta
) {
  auto delta_x = Fixed<F,T>::make(Fixed<F,T>::HALF >> log_width);
  auto delta_y = Fixed<F,T>::make(Fixed<F,T>::HALF >> log_height);

  uint32_t u0 = Clamp(fu - delta_x, wrapu);    
  uint32_t u1 = Clamp(fu + delta_x, wrapu);
  uint32_t v0 = Clamp(fv - delta_y, wrapv);     
  uint32_t v1 = Clamp(fv + delta_y, wrapv);

  uint32_t shift_u = (Fixed<F,T>::FRAC - log_width);
  uint32_t shift_v = (Fixed<F,T>::FRAC - log_height);

  uint32_t x0s = (u0 << 8) >> shift_u;
  uint32_t y0s = (v0 << 8) >> shift_v;

  uint32_t x0 = x0s >> 8;
  uint32_t y0 = y0s >> 8;
  uint32_t x1 = u1 >> shift_u;
  uint32_t y1 = v1 >> shift_v;

  *addr00 = x0 + (y0 << log_width);
  *addr01 = x1 + (y0 << log_width);
  *addr10 = x0 + (y1 << log_width);
  *addr11 = x1 + (y1 << log_width);

  *alpha  = x0s & 0xff;
  *beta   = y0s & 0xff;

  //printf("*** fu=0x%x, fv=0x%x, u0=0x%x, u1=0x%x, v0=0x%x, v1=0x%x, x0=0x%x, x1=0x%x, y0=0x%x, y1=0x%x, addr00=0x%x, addr01=0x%x, addr10=0x%x, addr11=0x%x\n", fu.data(), fv.data(), u0, u1, v0, v1, x0, x1, y0, y1, *addr00, *addr01, *addr10, *addr11);
}

template <uint32_t F, typename T = int32_t>
void TexAddressPoint(Fixed<F,T> fu, 
                     Fixed<F,T> fv, 
                     uint32_t log_width,
                     uint32_t log_height,
                     WrapMode wrapu,
                     WrapMode wrapv,
                     uint32_t* addr
) {
  uint32_t u = Clamp(fu, wrapu);
  uint32_t v = Clamp(fv, wrapv);
  
  uint32_t x = u >> (Fixed<F,T>::FRAC - log_width);
  uint32_t y = v >> (Fixed<F,T>::FRAC - log_height);
  
  *addr = x + (y << log_width);

  //printf("*** fu=0x%x, fv=0x%x, u=0x%x, v=0x%x, x=0x%x, y=0x%x, addr=0x%x\n", fu.data(), fv.data(), u, v, x, y, *addr);
}

inline uint32_t TexFilterLinear(
  TexFormat format,
  uint32_t texel00,  
  uint32_t texel01,
  uint32_t texel10,
  uint32_t texel11,
  uint32_t alpha,
  uint32_t beta
) {
  uint32_t c01l, c01h;
  {
    uint32_t c0l, c0h;  
    uint32_t c1l, c1h;
    Unpack8888(format, texel00, &c0l, &c0h);
    Unpack8888(format, texel01, &c1l, &c1h);
    Lerp8888(c0l, c0h, c1l, c1h, alpha, &c01l, &c01h);
  }

  uint32_t c23l, c23h;
  {
    uint32_t c2l, c2h;  
    uint32_t c3l, c3h;
    Unpack8888(format, texel10, &c2l, &c2h);
    Unpack8888(format, texel11, &c3l, &c3h);
    Lerp8888(c2l, c2h, c3l, c3h, alpha, &c23l, &c23h);
  }

  uint32_t cl, ch;
  Lerp8888(c01l, c01h, c23l, c23h, beta, &cl, &ch);
  uint32_t color = Pack8888(TexFormat::R8G8B8A8, cl, ch);

  //printf("*** texel00=0x%x, texel01=0x%x, texel10=0x%x, texel11=0x%x, color=0x%x\n", texel00, texel01, texel10, texel11, color);

  return color;
}

inline uint32_t TexFilterPoint(TexFormat format, uint32_t texel) {
  uint32_t cl, ch;  
  Unpack8888(format, texel, &cl, &ch);
  uint32_t color = Pack8888(TexFormat::R8G8B8A8, cl, ch);

  //printf("*** texel=0x%x, color=0x%x\n", texel, color);

  return color;
}