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

`ifndef VX_FPU_PKG_VH
`define VX_FPU_PKG_VH

`include "VX_define.vh"

package VX_fpu_pkg;

typedef struct packed {
    logic is_normal;
    logic is_zero;
    logic is_subnormal;
    logic is_inf;
    logic is_nan;
    logic is_quiet;
    logic is_signaling;    
} fclass_t;

typedef struct packed {
    logic NV; // 4-Invalid
    logic DZ; // 3-Divide by zero
    logic OF; // 2-Overflow
    logic UF; // 1-Underflow
    logic NX; // 0-Inexact
} fflags_t;

endpackage

`endif // VX_FPU_PKG_VH
