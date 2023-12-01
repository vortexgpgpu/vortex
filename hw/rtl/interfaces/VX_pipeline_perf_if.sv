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

interface VX_pipeline_perf_if ();
    wire [`PERF_CTR_BITS-1:0]   sched_stalls;
    wire [`PERF_CTR_BITS-1:0]   fetch_stalls;
    wire [`PERF_CTR_BITS-1:0]   ibf_stalls;
    wire [`PERF_CTR_BITS-1:0]   scb_stalls;
    wire [`PERF_CTR_BITS-1:0]   scb_uses [`NUM_EX_UNITS];
    wire [`PERF_CTR_BITS-1:0]   dsp_stalls [`NUM_EX_UNITS];

    wire [`PERF_CTR_BITS-1:0]   ifetches;
    wire [`PERF_CTR_BITS-1:0]   loads;
    wire [`PERF_CTR_BITS-1:0]   stores;    
    wire [`PERF_CTR_BITS-1:0]   ifetch_latency;
    wire [`PERF_CTR_BITS-1:0]   load_latency;

    modport schedule (
        output sched_stalls,
        output fetch_stalls
    );

    modport issue (
        output ibf_stalls,
        output scb_stalls,
        output scb_uses,
        output dsp_stalls
    );

    modport slave (
        input sched_stalls,
        input fetch_stalls,
        input ibf_stalls,
        input scb_stalls,
        input scb_uses,
        input dsp_stalls,
        input ifetches,
        input loads,
        input stores,
        input ifetch_latency,
        input load_latency
    );

endinterface
