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
    cache_perf_t smem;
    mem_perf_t   mem;

    modport master (
        output icache,
        output dcache,
        output l2cache,
        output l3cache,
        output smem,
        output mem
    );

    modport slave (
        input icache,
        input dcache,
        input l2cache,
        input l3cache,
        input smem,        
        input mem
    );

endinterface
