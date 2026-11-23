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

// TCU -> barrier unit: tc.commit arrivals.
interface VX_tcu_bar_if import VX_gpu_pkg::*; ();

`ifdef VX_CFG_TCU_TMEM_ENABLE
    logic                  req_valid;
    tcu_bar_t              req_data;
    logic                  req_ready;

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
`endif

endinterface
