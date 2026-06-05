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

// VX_rtu_node_decode — combinational extraction of a CW-BVH internal node
// from one fetched cache line into the width-generic rtu_node_t view. A
// CW-BVH4 node is exactly 64 B (one line); fields are sliced by the byte
// offsets in VX_rtu_pkg. Little-endian byte order: a uint32 at byte b is
// line[b*8 +: 32]. The leaf/internal kind tag and child count occupy
// word0 (kind in bits 0..7, num_children in bits 8..15).

`include "VX_define.vh"

module VX_rtu_node_decode import VX_rtu_pkg::*; #(
    parameter LINE_BITS = `VX_CFG_MEM_BLOCK_SIZE * 8
) (
    input  wire [LINE_BITS-1:0] line,
    output wire [7:0]           kind,       // node-kind tag (RTU_KIND_*)
    output rtu_node_t           node
);
    // word0: kind in [7:0], raw child/prim count in [15:8].
    wire [7:0] raw_count = line[(RTU_NODE_OFF_KIND*8 + 8) +: 8];
    assign kind = line[RTU_NODE_OFF_KIND*8 +: 8];

    // Clamp the declared count to the configured fan-out.
    assign node.n_children = (raw_count > 8'(RTU_BVH_WIDTH))
                           ? RTU_CHILD_BITS'(RTU_BVH_WIDTH)
                           : raw_count[RTU_CHILD_BITS-1:0];

    for (genvar a = 0; a < 3; ++a) begin : g_common
        assign node.origin[a] = line[(RTU_NODE_OFF_ORIGIN + 4*a)*8 +: 32];
        assign node.exp[a]    = line[(RTU_NODE_OFF_EXP + a)*8 +: 8];
    end

    for (genvar i = 0; i < RTU_BVH_WIDTH; ++i) begin : g_child
        assign node.child_off[i] = line[(RTU_NODE_OFF_CHILD + 4*i)*8 +: 32];
        for (genvar a = 0; a < 3; ++a) begin : g_axis
            assign node.qmin[i][a] = line[(RTU_NODE_OFF_QMIN + 3*i + a)*8 +: 8];
            assign node.qmax[i][a] = line[(RTU_NODE_OFF_QMAX + 3*i + a)*8 +: 8];
        end
    end

endmodule
