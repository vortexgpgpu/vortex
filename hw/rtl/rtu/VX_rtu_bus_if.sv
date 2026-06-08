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

// VX_rtu_bus_if — per-core SFU shim ↔ cluster-shared RTU core channel.
// The request carries the active-lane mask and per-lane ray snapshot taken
// from the ray-state RF at vx_rt_trace; the response carries the per-lane
// terminal status and hit attributes written back at vx_rt_wait. Both
// directions use valid/ready so the cluster arbiter can backpressure.
//
// Phase 2 (shader callbacks) overloads both directions with a kind tag:
//   req.kind = TRACE  — a fresh ray snapshot (rays); the common case.
//            = CBACT  — vx_rt_cb_ret releasing a parked context with a
//                       per-lane action (cb_action); rays unused.
//   rsp.kind = TERMINAL — slot finished (HIT/MISS); status + hit attrs.
//            = CBYIELD  — slot yielded a candidate mid-walk to a shader;
//                       cb_active_mask marks the yielding lanes, cb_type
//                       selects the dispatcher (AHS/IS/...), cb_sbt_idx
//                       keys the SBT, and the candidate attrs ride the
//                       same hit_* fields (staged into the regfile so the
//                       dispatcher reads them register-fast).

interface VX_rtu_bus_if import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) ();
    typedef struct packed {
        logic                                        kind;     // RTU_REQ_*
        logic [NUM_LANES-1:0]                         mask;
        rtu_ray_t [NUM_LANES-1:0]                     rays;     // TRACE only
        logic [NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] cb_action;// CBACT only
        logic [NUM_LANES-1:0][31:0]                   cb_hit_t; // CBACT: IS-computed t (PROC accept)
        logic [TAG_WIDTH-1:0]                         tag;
    } req_data_t;

    typedef struct packed {
        logic                                        kind;     // RTU_RSP_*
        logic [NUM_LANES-1:0][31:0]      status;          // VX_RT_STS_* (TERMINAL)
        logic [NUM_LANES-1:0][31:0]      hit_t;           // candidate t on CBYIELD
        logic [NUM_LANES-1:0][31:0]      hit_u;
        logic [NUM_LANES-1:0][31:0]      hit_v;
        logic [NUM_LANES-1:0][31:0]      hit_prim_id;
        logic [NUM_LANES-1:0][31:0]      hit_geometry;
        // CBYIELD only — yielding-lane mask + per-lane callback metadata.
        logic [NUM_LANES-1:0]                         cb_active_mask;
        logic [NUM_LANES-1:0][RTU_CB_TYPE_BITS-1:0]   cb_type;
        logic [NUM_LANES-1:0][RTU_CB_SBT_BITS-1:0]    cb_sbt_idx;
        logic [TAG_WIDTH-1:0]            tag;
    } rsp_data_t;

    logic       req_valid;
    req_data_t  req_data;
    logic       req_ready;

    logic       rsp_valid;
    rsp_data_t  rsp_data;
    logic       rsp_ready;

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
