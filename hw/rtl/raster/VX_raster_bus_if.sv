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

`include "VX_raster_define.vh"

interface VX_raster_bus_if import VX_raster_pkg::*; #(
    parameter NUM_LANES = 1
) ();
    typedef struct packed {
        raster_stamp_t [NUM_LANES-1:0]  stamps;
        logic                           done;
    } req_data_t;

    logic       req_valid;
    req_data_t  req_data;
    logic       req_ready;

    modport master (
        output req_valid,
        output req_data,
        input  req_ready
    );

    modport slave (
        input  req_valid,
        input  req_data,
        output req_ready
    );

endinterface
