#pragma once

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 3
#endif

#define DEBUG_HEADER << "DEBUG "
//#define DEBUG_HEADER << "DEBUG " << __FILE__ << ':' << std::dec << __LINE__ << ": "

#define TRACE_HEADER << "TRACE "
//#define TRACE_HEADER << "DEBUG " << __FILE__ << ':' << std::dec << __LINE__ << ": "

#ifndef NDEBUG

#include <iostream>
#include <iomanip>

#define DP(lvl, x) do { \
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

#define DT(lvl, x) do { \
  if ((lvl) <= DEBUG_LEVEL) { \
    std::cout TRACE_HEADER << std::setw(10) << std::dec << SimPlatform::instance().cycles() << std::setw(0) << ": " << x << std::endl; \
  } \
} while(0)

#define DTH(lvl, x) do { \
  if ((lvl) <= DEBUG_LEVEL) { \
    std::cout TRACE_HEADER << std::setw(10) << std::dec << SimPlatform::instance().cycles() << std::setw(0) << ": " << x; \
  } \
} while(0)

#define DTN(lvl, x) do { \
  if ((lvl) <= DEBUG_LEVEL) { \
    std::cout << x; \
  } \
} while(0)


#else

#define DP(lvl, x) do {} while(0)
#define DPH(lvl, x) do {} while(0)
#define DPN(lvl, x) do {} while(0)

#define DT(lvl, x) do {} while(0)
#define DTH(lvl, x) do {} while(0)
#define DTN(lvl, x) do {} while(0)

#endif