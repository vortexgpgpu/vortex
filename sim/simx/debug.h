// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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