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

// Generic SFU side CSR interface — used by per-extension CSR modules
// (raster, tex, om) to expose per-warp + per-pid CSR data to VX_csr_unit.
// `read_data` is per-lane so different lanes within a warp can return
// different values (e.g., raster bcoords differ per quad corner).
//
// The write side carries kernel-issued CSR writes (CSRRW / CSRRS / CSRRC).
// For raster CSRs (read-only state latched on vx_rast pop) these are
// unused; tex/om may use them to receive runtime writes.
interface VX_sfu_csr_if import VX_gpu_pkg::*; #(
    parameter NUM_LANES = `NUM_THREADS,
    parameter PID_WIDTH = `LOG2UP(`NUM_THREADS / NUM_LANES)
) ();

    // Read interface
    wire                            read_enable;
    wire [`VX_CSR_ADDR_BITS-1:0]    read_addr;
    wire [UUID_WIDTH-1:0]           read_uuid;
    wire [NW_WIDTH-1:0]            read_wid;
    wire [`UP(PID_WIDTH)-1:0]       read_pid;
    wire [NUM_LANES-1:0]            read_tmask;
    wire [NUM_LANES-1:0][`XLEN-1:0] read_data;

    // Write interface (kernel-issued CSRRW/RS/RC; unused for read-only CSRs)
    wire                            write_enable;
    wire [`VX_CSR_ADDR_BITS-1:0]    write_addr;
    wire [`XLEN-1:0]                write_data;
    wire [UUID_WIDTH-1:0]           write_uuid;
    wire [NW_WIDTH-1:0]            write_wid;
    wire [`UP(PID_WIDTH)-1:0]       write_pid;
    wire [NUM_LANES-1:0]            write_tmask;

    modport master (
        output read_enable, read_addr, read_uuid, read_wid, read_pid, read_tmask,
        input  read_data,
        output write_enable, write_addr, write_data, write_uuid, write_wid, write_pid, write_tmask
    );

    modport slave (
        input  read_enable, read_addr, read_uuid, read_wid, read_pid, read_tmask,
        output read_data,
        input  write_enable, write_addr, write_data, write_uuid, write_wid, write_pid, write_tmask
    );

endinterface
