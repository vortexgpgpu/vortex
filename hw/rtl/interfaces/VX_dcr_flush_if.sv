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

// Handshake interface for a DCR-triggered cache flush.
// req:  asserted by the initiator to trigger a full cache flush.
// done: pulsed by the cache for one cycle when all banks have completed flushing.
interface VX_dcr_flush_if ();

  wire req;
  wire done;

  modport master (output req, input  done);
  modport slave  (input  req, output done);

endinterface
