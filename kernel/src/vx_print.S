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
#include "common.h"

.type vx_putchar, @function
.global vx_putchar
vx_putchar:
    csrr t0, VX_CSR_MHARTID
    andi t0, t0, %lo(IO_COUT_SIZE-1)
    li t1, IO_COUT_ADDR
    add t0, t0, t1
    sb a0, 0(t0)
    ret
