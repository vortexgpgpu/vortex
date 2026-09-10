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

// Level-held TLB flush handshake: the DCR surface asserts req and every
// TLB/PTW leg holds done until it has invalidated its state and drained
// in-flight walks; the AND-tree over all done legs signals completion.
interface VX_tlb_flush_if ();

    logic req;
    logic done;

    modport master (
        output req,
        input  done
    );

    modport slave (
        input  req,
        output done
    );

endinterface
