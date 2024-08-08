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

#define RISCV_CUSTOM0   0x0B

.type vx_serial, @function
.global vx_serial
vx_serial:
#if (XLEN == 64)
    addi sp, sp, -56
    sd   ra, 48(sp)
    sd   s5, 40(sp)
    sd   s4, 32(sp)
    sd   s3, 24(sp)
    sd   s2, 16(sp)
    sd   s1, 8(sp)
    sd   s0, 0(sp)
#else
    addi sp, sp, -28
    sw   ra, 24(sp)
    sw   s5, 20(sp)
    sw   s4, 16(sp)
    sw   s3, 12(sp)
    sw   s2, 8(sp)
    sw   s1, 4(sp)
    sw   s0, 0(sp)
#endif
    mv   s4, a0                 # s4 <- callback
    mv   s3, a1                 # s3 <- arg
    csrr s2, VX_CSR_NUM_THREADS # s2 <- NT
    csrr s1, VX_CSR_THREAD_ID   # s1 <- tid
    li	 s0, 0                  # s0 <- index
label_loop:
    sub	 t0, s0, s1
    seqz t1, t0                 # (index != tid)
    .insn r RISCV_CUSTOM0, 2, 0, s5, t1, x0  # split s5, t0
    bnez t0, label_join
    mv   a0, s3                 # a0 <- arg
    jalr s4                     # callback(arg)
label_join:
    .insn r RISCV_CUSTOM0, 3, 0, x0, s5, x0  # join s5
    addi s0, s0, 1              # index++
    blt	 s0, s2, label_loop     # loop back
#if (XLEN == 64)
    ld   ra, 48(sp)
    ld   s5, 40(sp)
    ld   s4, 32(sp)
    ld   s3, 24(sp)
    ld   s2, 16(sp)
    ld   s1, 8(sp)
    ld   s0, 0(sp)
    addi sp, sp, 56
#else
    lw   ra, 24(sp)
    lw   s5, 20(sp)
    lw   s4, 16(sp)
    lw   s3, 12(sp)
    lw   s2, 8(sp)
    lw   s1, 4(sp)
    lw   s0, 0(sp)
    addi sp, sp, 28
#endif
    ret
