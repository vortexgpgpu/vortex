#pragma once

//#define USE_DEBUG 9

#ifdef USE_DEBUG

#include <iostream>
#include <iomanip>

#define DX(x) x

#define D(lvl, x) do { \
  if ((lvl) <= USE_DEBUG) { \
    std::cout << "DEBUG " << __FILE__ << ':' << std::dec << __LINE__ << ": " << x << std::endl; \
  } \
} while(0)

#define DPH(lvl, x) do { \
  if ((lvl) <= USE_DEBUG) { \
    std::cout << "DEBUG " << __FILE__ << ':' << std::dec << __LINE__ << ": " << x; \
  } \
} while(0)

#define DPN(lvl, x) do { \
  if ((lvl) <= USE_DEBUG) { \
    std::cout << x; \
  } \
} while(0)

#define D_RAW(x) do { \
  std::cout << x; \
} while (0)

#else

#define DX(x)
#define D(lvl, x) do {} while(0)
#define DPH(lvl, x) do {} while(0)
#define DPN(lvl, x) do {} while(0)
#define D_RAW(x) do {} while(0)

#endif