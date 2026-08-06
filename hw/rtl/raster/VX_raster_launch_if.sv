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

// KMU → raster-engine delegated-launch kick: a payload-less single-token
// stream with standard valid/ready semantics (one launch outstanding at a
// time — the kernel-launch fence serializes frames). Distributed by an
// eager VX_stream_fork (VX_raster_launch_fork), so the master's handshake
// completes only when every engine has consumed the kick — the KMU holds
// the device busy across delivery and each engine holds it from its own
// acceptance, leaving the launch fence no busy gap to observe.
interface VX_raster_launch_if ();

    logic valid;
    logic ready;

    modport master (
        output valid,
        input  ready
    );

    modport slave (
        input  valid,
        output ready
    );

endinterface
