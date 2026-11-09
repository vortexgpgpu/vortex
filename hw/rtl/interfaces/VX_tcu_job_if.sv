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

// A TCU block <-> the core's tensor memory: the block's TMEM and tc ops and
// their responses, and the tensor-op MMA steps the block's core executes.
interface VX_tcu_job_if ();

`ifdef VX_CFG_TCU_TMEM_ENABLE
    import VX_gpu_pkg::*;
    import VX_tcu_pkg::*;

    logic           op_valid;
    tcu_execute_t   op_data;
    logic           op_ready;

    logic           rsp_valid;
    tcu_result_t    rsp_data;
    logic           rsp_ready;

    // MMA steps, issued to every block in the same cycle
    logic           job_valid;
    logic           job_fire;
    tcu_execute_t   job_data;
    logic           core_ready;

    logic           jres_valid;
    logic [`VX_CFG_NUM_THREADS-1:0][31:0] jres_data;

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    logic [TCU_MAX_META_BLOCK_WIDTH-1:0] job_meta;
    logic [TCU_MAX_META_BLOCK_WIDTH-1:0] job_meta_hi;
`endif

    modport block (
        output op_valid,
        output op_data,
        input  op_ready,
        input  rsp_valid,
        input  rsp_data,
        output rsp_ready,
        input  job_valid,
        input  job_fire,
        input  job_data,
        output core_ready,
        output jres_valid,
        output jres_data
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        ,
        input  job_meta,
        input  job_meta_hi
    `endif
    );

    modport tmem (
        input  op_valid,
        input  op_data,
        output op_ready,
        output rsp_valid,
        output rsp_data,
        input  rsp_ready,
        output job_valid,
        output job_fire,
        output job_data,
        input  core_ready,
        input  jres_valid,
        input  jres_data
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        ,
        output job_meta,
        output job_meta_hi
    `endif
    );
`endif

endinterface
