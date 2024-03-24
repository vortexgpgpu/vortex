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
module VX_aos_to_soa #(
    parameter int NUM_FIELDS = 1,
    parameter int FIELD_WIDTHS[NUM_FIELDS] = '{1},
    parameter int NUM_ELEMENTS = 100,
    parameter int STRUCT_WIDTH = FIELD_WIDTHS[0]
) (
    input wire [(NUM_ELEMENTS*STRUCT_WIDTH)-1:0] aos_in,
    output wire [(NUM_ELEMENTS*STRUCT_WIDTH)-1:0] soa_out
);
    int field_offsets[NUM_FIELDS];
    initial begin
        field_offsets[0] = 0;
        for (int i = 1; i < NUM_FIELDS; i++) begin
            field_offsets[i] = field_offsets[i-1] + FIELD_WIDTHS[i-1];
        end
    end

    for (genvar i = 0; i < NUM_ELEMENTS; i++) begin : elems
        for (genvar j = 0; j < NUM_FIELDS; j++) begin : fields
        `IGNORE_UNUSED_BEGIN
            int soa_offset = j * NUM_ELEMENTS * field_offsets[j] + i * FIELD_WIDTHS[j];
            int aos_offset = i * STRUCT_WIDTH + field_offsets[j];
        `IGNORE_UNUSED_END
            assign soa_out[soa_offset +: FIELD_WIDTHS[j]] = aos_in[aos_offset +: FIELD_WIDTHS[j]];
        end
    end

endmodule
`TRACING_ON
