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

// VX_kmu_bus_if — the device launch stream.
//
// A launch is a MESSAGE, not a beat. Beat 0 is always the kmu_req_t header; a
// fragment launch appends payload beats carrying the raster stamp packed at bus
// width. Payload beats reuse the header's wires — `data` is raw here and the
// endpoints cast it — so delivering the stamp inside the launch does not widen
// the bus. `kind` and `eop` are the entire interface change.
//
// The master must hold `valid` bubble-free from the header through `eop`: the
// arbiters lock an input->output pair for the length of a message and release it
// on `eop`, not on a lull. A mid-message bubble would let another master's
// header interleave into the same output stream.
//
// `dest` is the global core index a FRAGMENT launch must land on. Fragment work
// is bin->core affine — that affinity is what keeps same-pixel blend order
// correct — so a fragment message cannot be load-balanced the way a compute CTA
// can. Each fan-out level consumes its own slice of `dest` (KMU_DEST_LSB_*);
// compute launches ignore it and go to any ready output.

interface VX_kmu_bus_if import VX_gpu_pkg::*; ();
    logic                   valid;
    logic                   kind;   // KMU_KIND_COMPUTE | KMU_KIND_FRAGMENT
    logic                   eop;    // last beat of the message
    logic [KMU_DEST_W-1:0]  dest;   // placement hint (fragment only)
    logic [KMU_DATAW-1:0]   data;   // beat 0 = kmu_req_t; payload beats = packed stamps
    logic                   ready;

    modport master (
        output valid,
        output kind,
        output eop,
        output dest,
        output data,
        input  ready
    );

    modport slave (
        input  valid,
        input  kind,
        input  eop,
        input  dest,
        input  data,
        output ready
    );

endinterface
