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

`ifndef VX_FPU_DEFINE_VH
`define VX_FPU_DEFINE_VH

`include "VX_define.vh"

`ifdef SV_DPI
`include "float_dpi.vh"
`endif

`define FPU_MERGE_FFLAGS(out, in, mask, lanes) \
    fflags_t __``out; \
    always @(*) begin \
        __``out = '0; \
        for (integer __i = 0; __i < lanes; ++__i) begin \
            if (mask[__i]) begin \
                __``out.NX |= in[__i].NX; \
                __``out.UF |= in[__i].UF; \
                __``out.OF |= in[__i].OF; \
                __``out.DZ |= in[__i].DZ; \
                __``out.NV |= in[__i].NV; \
            end \
        end \
    end \
    assign out = __``out
    
`define FP_CLASS_BITS   $bits(VX_fpu_pkg::fclass_t)
`define FP_FLAGS_BITS   $bits(VX_fpu_pkg::fflags_t)

`endif // VX_FPU_DEFINE_VH
