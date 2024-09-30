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

#include <VX_config.h>

#ifndef RAM_PAGE_SIZE
#define RAM_PAGE_SIZE     4096
#endif

#ifndef MEM_CLOCK_RATIO
#define MEM_CLOCK_RATIO   1
#endif

inline constexpr int LSU_WORD_SIZE    = (XLEN / 8);
inline constexpr int LSU_CHANNELS     = NUM_LSU_LANES;
inline constexpr int LSU_NUM_REQS	    = (NUM_LSU_BLOCKS * LSU_CHANNELS);

inline constexpr int DCACHE_WORD_SIZE = LSU_LINE_SIZE;
inline constexpr int DCACHE_CHANNELS 	= UP((NUM_LSU_LANES * (XLEN / 8)) / DCACHE_WORD_SIZE);
inline constexpr int DCACHE_NUM_REQS	=  (NUM_LSU_BLOCKS * DCACHE_CHANNELS);

inline constexpr int NUM_SOCKETS      = UP(NUM_CORES / SOCKET_SIZE);

inline constexpr int PER_ISSUE_WARPS  = NUM_WARPS / ISSUE_WIDTH;