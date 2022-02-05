#pragma once

#include <cstdint>
#include <cocogfx/include/fixed.h>
#include <bitmanip.h>

using namespace cocogfx;

enum class WrapMode {
  Clamp,
  Repeat,
  Mirror,
};

enum class TexFormat {
  A8R8G8B8,
  R5G6B5,
  A1R5G5B5,
  A4R4G4B4,
  A8L8,
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
  case TexFormat::A8R8G8B8: 
    return 4;
  case TexFormat::R5G6B5:
  case TexFormat::A1R5G5B5:
  case TexFormat::A4R4G4B4:
  case TexFormat::A8L8:
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
  uint32_t r, g, b, a;
  switch (format) {
  case TexFormat::A8R8G8B8:    
    r = (texel >> 16) & 0xff;
    g = (texel >> 8) & 0xff;
    b = texel & 0xff;
    a = texel >> 24;
    break;
  case TexFormat::R5G6B5: 
    r = ((texel >> 11) << 3) | (texel >> 13);    
    g = ((texel >> 3) & 0xfc) | ((texel >> 9) & 0x3);
    b = ((texel & 0x1f) << 3) | ((texel & 0x1c) >> 2);    
    a = 0xff;
    break;
  case TexFormat::A1R5G5B5:         
    r = ((texel >> 7) & 0xf8) | ((texel << 1) >> 13);
    g = ((texel >> 2) & 0xf8) | ((texel >> 7) & 7);
    b = ((texel & 0x1f) << 3) | ((texel & 0x1c) >> 2);
    a = 0xff * (texel >> 15);
    break;
  case TexFormat::A4R4G4B4:   
    r = ((texel >> 4) & 0xf0) | ((texel >> 8) & 0x0f);
    g = ((texel & 0xf0) >> 0) | ((texel & 0xf0) >> 4);
    b = ((texel & 0x0f) << 4) | ((texel & 0x0f) >> 0);
    a = ((texel >> 8) & 0xf0) | (texel >> 12);
    break;
  case TexFormat::A8L8:
    r = texel & 0xff;
    g = r;
    b = r;
    a = texel >> 8;
    break;
  case TexFormat::L8:
    r = texel & 0xff;
    g = r;
    b = r;
    a = 0xff;
    break;
  case TexFormat::A8:
    r = 0xff;
    g = 0xff;
    b = 0xff;
    a = texel & 0xff;
    break;
  default: 
    std::abort();
  } 
  *lo = (r << 16) + b;
  *hi = (a << 16) + g;
}

inline void Unpack8888(uint32_t texel, uint32_t* lo, uint32_t* hi) {
  *lo = texel & 0x00ff00ff;
  *hi = (texel >> 8) & 0x00ff00ff;
}

inline uint32_t Pack8888(uint32_t lo, uint32_t hi) {
  return (hi << 8) | lo;
}

inline uint32_t Lerp8888(uint32_t a, uint32_t b, uint32_t f) {
  return (a + (((b - a) * f) >> 8)) & 0x00ff00ff;
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
    uint32_t c0l, c0h, c1l, c1h;
    Unpack8888(format, texel00, &c0l, &c0h);
    Unpack8888(format, texel01, &c1l, &c1h);
    c01l = Lerp8888(c0l, c1l, alpha);
    c01h = Lerp8888(c0h, c1h, alpha);
  }

  uint32_t c23l, c23h;
  {
    uint32_t c2l, c2h, c3l, c3h;
    Unpack8888(format, texel10, &c2l, &c2h);
    Unpack8888(format, texel11, &c3l, &c3h);
    c23l = Lerp8888(c2l, c3l, alpha);
    c23h = Lerp8888(c2h, c3h, alpha);
  }

  uint32_t color;
  {
    uint32_t cl = Lerp8888(c01l, c23l, beta);
    uint32_t ch = Lerp8888(c01h, c23h, beta);
    color = Pack8888(cl, ch);
  }

  //printf("*** texel00=0x%x, texel01=0x%x, texel10=0x%x, texel11=0x%x, color=0x%x\n", texel00, texel01, texel10, texel11, color);

  return color;
}

inline uint32_t TexFilterPoint(TexFormat format, uint32_t texel) {
  uint32_t color;
  {
    uint32_t cl, ch;
    Unpack8888(format, texel, &cl, &ch);
    color = Pack8888(cl, ch);
  }

  //printf("*** texel=0x%x, color=0x%x\n", texel, color);

  return color;
}