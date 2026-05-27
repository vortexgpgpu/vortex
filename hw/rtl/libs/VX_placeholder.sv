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

`include "VX_platform.vh"

`TRACING_OFF
`BLACKBOX_CELL module VX_placeholder #(
    parameter I = 0,
    parameter O = 0
) (
    input wire [`UP(I)-1:0] in,
    output wire [`UP(O)-1:0] out
);
    // empty module

endmodule
`TRACING_ON
