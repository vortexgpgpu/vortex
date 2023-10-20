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

`include "VX_define.vh"

interface VX_gbar_bus_if ();

    wire                    req_valid;
    wire [`NB_WIDTH-1:0]    req_id;
    wire [`NC_WIDTH-1:0]    req_size_m1;
    wire [`NC_WIDTH-1:0]    req_core_id;
    wire                    req_ready;

    wire                    rsp_valid;
    wire [`NB_WIDTH-1:0]    rsp_id;

    modport master (
        output  req_valid,
        output  req_id,
        output  req_size_m1,    
        output  req_core_id,
        input   req_ready,

        input   rsp_valid,
        input   rsp_id
    );

    modport slave (
        input   req_valid,
        input   req_id,
        input   req_size_m1,
        input   req_core_id,
        output  req_ready,
        
        output  rsp_valid,
        output  rsp_id
    );

endinterface
