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

#include <VX_config.h>
#include <VX_types.h>

.section .init, "ax"
.global _start
.type _start, @function
_start:
  # call main routine
  call main

  # dump perf counter
  call vx_perf_dump

  # end execution
  .insn r 0x0b, 0, 0, x0, x0, x0
.size _start, .-_start