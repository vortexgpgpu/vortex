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

`include "VX_fpu_define.vh"

interface VX_fpu_csr_if import VX_fpu_pkg::*; ();

    wire                    write_enable;
    wire [`NW_WIDTH-1:0]    write_wid;
    fflags_t                write_fflags;

    wire [`NW_WIDTH-1:0]    read_wid;
    wire [`INST_FRM_BITS-1:0] read_frm;

    modport master (
        output write_enable,
        output write_wid,
        output write_fflags,

        output read_wid,
        input  read_frm
    );

    modport slave (
        input  write_enable,
        input  write_wid,
        input  write_fflags,
        
        input  read_wid,
        output read_frm
    );

endinterface
