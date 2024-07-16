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

interface VX_mem_perf_if import VX_gpu_pkg::*; ();

    cache_perf_t icache;
    cache_perf_t dcache;
    cache_perf_t l2cache;
    cache_perf_t l3cache;
`ifdef EXT_TEX_ENABLE
    cache_perf_t tcache;
`endif
`ifdef EXT_RASTER_ENABLE
    cache_perf_t rcache;
`endif
`ifdef EXT_OM_ENABLE
    cache_perf_t ocache;
`endif
    cache_perf_t lmem;
    mem_perf_t   mem;

    modport master (
        output icache,
        output dcache,
        output l2cache,
        output l3cache,
    `ifdef EXT_TEX_ENABLE
        output tcache,
    `endif
    `ifdef EXT_RASTER_ENABLE
        output rcache,
    `endif
    `ifdef EXT_OM_ENABLE
        output ocache,
    `endif
        output lmem,
        output mem
    );

    modport slave (
        input icache,
        input dcache,
        input l2cache,
        input l3cache,
    `ifdef EXT_TEX_ENABLE
        input tcache,
    `endif
    `ifdef EXT_RASTER_ENABLE
        input rcache,
    `endif
    `ifdef EXT_OM_ENABLE
        input ocache,
    `endif
        input lmem,
        input mem
    );

endinterface
