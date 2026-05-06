//!/bin/bash

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

`include "VX_raster_define.vh"

module VX_raster_extents #(
    parameter TILE_LOGSIZE = 5
) (
    input wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges,
    output wire [2:0][`RASTER_DATA_BITS-1:0]     extents
);
    for (genvar i = 0; i < 3; ++i) begin : g_extents
        wire [`RASTER_DATA_BITS-1:0] edge_x_m = {`RASTER_DATA_BITS{~edges[i][0][`RASTER_DATA_BITS-1]}};
        wire [`RASTER_DATA_BITS-1:0] edge_y_m = {`RASTER_DATA_BITS{~edges[i][1][`RASTER_DATA_BITS-1]}};
        assign extents[i] = (edge_x_m & (edges[i][0] << TILE_LOGSIZE))
                          + (edge_y_m & (edges[i][1] << TILE_LOGSIZE));
    end

endmodule
