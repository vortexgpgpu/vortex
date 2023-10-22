//!/bin/bash

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

`include "VX_tex_define.vh"

interface VX_tex_bus_if #(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) ();
    typedef struct packed {
        logic [NUM_LANES-1:0]            mask;
        logic [1:0][NUM_LANES-1:0][31:0] coords;
        logic [NUM_LANES-1:0][`VX_TEX_LOD_BITS-1:0] lod;
        logic [`VX_TEX_STAGE_BITS-1:0]   stage;
        logic [TAG_WIDTH-1:0]            tag;  
    } req_data_t;

    typedef struct packed {
        logic [NUM_LANES-1:0][31:0]      texels;
        logic [TAG_WIDTH-1:0]            tag; 
    } rsp_data_t;

    logic       req_valid;
    req_data_t  req_data;
    logic       req_ready;

    logic  rsp_valid;
    rsp_data_t rsp_data;
    logic  rsp_ready;

    modport master (
        output req_valid,
        output req_data,
        input  req_ready,

        input  rsp_valid,
        input  rsp_data,
        output rsp_ready
    );

    modport slave (
        input  req_valid,
        input  req_data,
        output req_ready,

        output rsp_valid,
        output rsp_data,
        input  rsp_ready
    );

endinterface
