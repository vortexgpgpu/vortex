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

interface VX_cache_perf_if ();

    wire [`PERF_CTR_BITS-1:0] reads;
    wire [`PERF_CTR_BITS-1:0] writes;
    wire [`PERF_CTR_BITS-1:0] read_misses;
    wire [`PERF_CTR_BITS-1:0] write_misses;
    wire [`PERF_CTR_BITS-1:0] bank_stalls;
    wire [`PERF_CTR_BITS-1:0] mshr_stalls;
    wire [`PERF_CTR_BITS-1:0] mem_stalls;
    wire [`PERF_CTR_BITS-1:0] crsp_stalls;

    modport master (
        output reads,
        output writes,
        output read_misses,
        output write_misses,
        output bank_stalls,
        output mshr_stalls,
        output mem_stalls,
        output crsp_stalls
    );

    modport slave (
        input reads,
        input writes,
        input read_misses,
        input write_misses,
        input bank_stalls,
        input mshr_stalls,
        input mem_stalls,
        input crsp_stalls
    );

endinterface
