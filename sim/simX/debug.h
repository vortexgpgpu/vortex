#pragma once

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 3
#endif

#define DEBUG_HEADER << "DEBUG "
//#define DEBUG_HEADER << "DEBUG " << __FILE__ << ':' << std::dec << __LINE__ << ": "

#ifndef NDEBUG

#include <iostream>
#include <iomanip>

#define DX(x) x

#define D(lvl, x) do { \
  if ((lvl) <= DEBUG_LEVEL) { \
    std::cout DEBUG_HEADER << x << std::endl; \
  } \
} while(0)

#define DPH(lvl, x) do { \
  if ((lvl) <= DEBUG_LEVEL) { \
    std::cout DEBUG_HEADER << x; \
  } \
} while(0)

#define DPN(lvl, x) do { \
  if ((lvl) <= DEBUG_LEVEL) { \
    std::cout << x; \
  } \
} while(0)

#else

#define DX(x)
#define D(lvl, x) do {} while(0)
#define DPH(lvl, x) do {} while(0)
#define DPN(lvl, x) do {} while(0)
#define D_RAW(x) do {} while(0)

#endif