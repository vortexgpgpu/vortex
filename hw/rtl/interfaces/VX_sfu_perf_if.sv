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

interface VX_sfu_perf_if ();

`ifdef EXT_TEX_ENABLE
    wire [`PERF_CTR_BITS-1:0] tex_stalls;
`endif
`ifdef EXT_RASTER_ENABLE
    wire [`PERF_CTR_BITS-1:0] raster_stalls;
`endif
`ifdef EXT_OM_ENABLE
    wire [`PERF_CTR_BITS-1:0] om_stalls;
`endif
    wire [`PERF_CTR_BITS-1:0] wctl_stalls;

    modport master (
    `ifdef EXT_TEX_ENABLE
        output tex_stalls,
    `endif
    `ifdef EXT_RASTER_ENABLE
        output raster_stalls,
    `endif
    `ifdef EXT_OM_ENABLE
        output om_stalls,
    `endif
        output wctl_stalls
    );

    modport slave (
    `ifdef EXT_TEX_ENABLE
        input tex_stalls,
    `endif
    `ifdef EXT_RASTER_ENABLE
        input raster_stalls,
    `endif
    `ifdef EXT_OM_ENABLE
        input om_stalls,
    `endif
        input wctl_stalls
    );

endinterface
