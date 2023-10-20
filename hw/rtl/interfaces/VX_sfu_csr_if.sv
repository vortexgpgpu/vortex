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

interface VX_sfu_csr_if  #(
    parameter NUM_LANES = `NUM_SFU_LANES,
    parameter PID_WIDTH = `LOG2UP(`NUM_THREADS / NUM_LANES)
) ();

    wire                        read_enable;
    wire [`UUID_WIDTH-1:0]      read_uuid;
    wire [`NW_WIDTH-1:0]        read_wid;
    wire [NUM_LANES-1:0]        read_tmask;
    wire [PID_WIDTH-1:0]        read_pid;
    wire [`VX_CSR_ADDR_BITS-1:0] read_addr;
    wire [NUM_LANES-1:0][31:0]  read_data;

    wire                        write_enable; 
    wire [`UUID_WIDTH-1:0]      write_uuid;
    wire [`NW_WIDTH-1:0]        write_wid;
    wire [NUM_LANES-1:0]        write_tmask;
    wire [PID_WIDTH-1:0]        write_pid;
    wire [`VX_CSR_ADDR_BITS-1:0] write_addr;
    wire [NUM_LANES-1:0][31:0]  write_data;

    modport master (
        output read_enable,
        output read_uuid,
        output read_wid,
        output read_tmask,
        output read_pid,
        output read_addr,
        input  read_data,

        output write_enable,
        output write_uuid,
        output write_wid,
        output write_tmask,
        output write_pid,
        output write_addr,
        output write_data
    );

    modport slave (
        input  read_enable,
        input  read_uuid,
        input  read_wid,
        input  read_tmask,
        input  read_pid,
        input  read_addr,
        output read_data,

        input  write_enable,
        input  write_uuid,
        input  write_wid,
        input  write_tmask,
        input  write_pid,
        input  write_addr,
        input  write_data
    );

endinterface
