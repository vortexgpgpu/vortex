`include "VX_raster_define.vh"

module VX_raster_req_switch #(  
    parameter CLUSTER_ID = 0,
    parameter RASTER_SLICE_NUM = 4,
    parameter RASTER_TILE_SIZE = 16,
    parameter RASTER_RS_SIZE = 8, // Reservation station size
    parameter RASTER_SLICE_BITS = `LOG2UP(RASTER_SLICE_NUM)
) (
    // Standard inputs
    input logic         clk,
    input logic         reset,

    // To indicate valid inputs provided
    input logic         input_valid,
    // Tile information
    input logic [`RASTER_DIM_BITS-1:0]        x_loc, y_loc,
    // Edge function values
    input logic [`RASTER_PRIMITIVE_DATA_BITS-1:0]   edge_func_val[2:0],
    // Memory information
    input logic [`RASTER_DCR_DATA_BITS-1:0]         mem_base_addr,
    input logic [`RASTER_DCR_DATA_BITS-1:0]         mem_stride,

    // Raster slice interactions
    input logic [RASTER_SLICE_NUM-1:0]                          raster_slice_ready,
    output logic [`RASTER_DIM_BITS-1:0]                   out_x_loc, out_y_loc,
    output logic [`RASTER_PRIMITIVE_DATA_BITS-1:0]              out_edges[2:0][2:0],
    output logic [`RASTER_PRIMITIVE_DATA_BITS-1:0]              out_edge_func_val[2:0],
    output logic [`RASTER_PRIMITIVE_DATA_BITS-1:0]              out_extents[2:0],
    output logic [RASTER_SLICE_BITS-1:0]                        out_slice_index,

    // Status signals
    output logic                                    ready
);

    // Holds x_loc, y_loc, edge_func_val, edges -> extents are calculated on the fly
    localparam RASTER_RS_DATA_WIDTH = 2*`RASTER_DIM_BITS + 3*`RASTER_PRIMITIVE_DATA_BITS + 3*3*`RASTER_PRIMITIVE_DATA_BITS;
    localparam RASTER_RS_INDEX_BITS = `LOG2UP(RASTER_RS_SIZE);

    // Reservation station
    logic [RASTER_RS_DATA_WIDTH-1:0]    raster_rs[RASTER_RS_SIZE-1:0];
    logic [RASTER_RS_SIZE-1:0]          raster_rs_valid;
    logic [RASTER_RS_SIZE-1:0]          raster_rs_empty;

    logic [RASTER_RS_INDEX_BITS-1:0]    raster_rs_empty_index, raster_rs_index;

    // Status signals
    logic valid_raster_index, valid_rs_index, valid_rs_empty_index;

    // Memory interactions
    logic mem_req_valid, mem_req_ready, mem_rsp_valid;
    logic [8:0][`RASTER_DCR_DATA_BITS-1:0]       mem_req_addr;
    logic [8:0][`RASTER_PRIMITIVE_DATA_BITS-1:0] mem_rsp_data;
    logic [RASTER_RS_INDEX_BITS-1:0]             raster_mem_rsp_tag;

    // Stall signal
    //  -> assert when any entry in the RS is empty
    assign ready = |raster_rs_empty & mem_req_ready;

    always @(posedge clk) begin
        // Reset condition
        if (reset) begin
            mem_req_valid <= 0;
            for (int i = 0; i < RASTER_RS_SIZE; ++i) begin
                raster_rs_valid[i] <= 0;
                raster_rs_empty[i] <= 1;
            end
        end
        else begin
            // On new input
            if (ready && input_valid && valid_rs_empty_index) begin
                raster_rs[raster_rs_empty_index] <= {x_loc, y_loc,
                    edge_func_val[0], edge_func_val[1], edge_func_val[2],
                    (3*3*`RASTER_PRIMITIVE_DATA_BITS)'(1'b0)};
                raster_rs_valid[raster_rs_empty_index] <= 0;
                raster_rs_empty[raster_rs_empty_index] <= 0;

                // Send the request to the memory stream reader
                mem_req_valid <= 1;
                for (int i = 0; i < 3; ++i) begin
                    for (int j = 0; j < 3; ++j) begin
                        mem_req_addr[i*3+j] <= mem_base_addr + (i*3+j)*mem_stride;
                    end
                end
            end
            else mem_req_valid <= 0;

            // When any raster slice is ready
            if (valid_raster_index && valid_rs_index) begin
                {out_x_loc, out_y_loc,
                out_edge_func_val[0], out_edge_func_val[1], out_edge_func_val[2],
                out_edges[0][0], out_edges[0][1], out_edges[0][2],
                out_edges[1][0], out_edges[1][1], out_edges[1][2],
                out_edges[2][0], out_edges[2][1], out_edges[2][2]} <= raster_rs[raster_rs_index];
                raster_rs_valid[raster_rs_index] <= 0;
                raster_rs_empty[raster_rs_index] <= 1;
            end

            if (mem_rsp_valid) begin
                // Memory response data
                raster_rs[raster_mem_rsp_tag][3*3*`RASTER_PRIMITIVE_DATA_BITS-1:0] <= {
                    mem_rsp_data[0], mem_rsp_data[1], mem_rsp_data[2],
                    mem_rsp_data[3], mem_rsp_data[4], mem_rsp_data[5],
                    mem_rsp_data[6], mem_rsp_data[7], mem_rsp_data[8]
                };
                raster_rs_valid[raster_mem_rsp_tag] <= 1;

                // Now if there is no other data ready but raster slice is free => forward
                if (valid_rs_index && valid_raster_index) begin
                    {out_x_loc, out_y_loc,
                    out_edge_func_val[0], out_edge_func_val[1], out_edge_func_val[2],
                    out_edges[0][0], out_edges[0][1], out_edges[0][2],
                    out_edges[1][0], out_edges[1][1], out_edges[1][2],
                    out_edges[2][0], out_edges[2][1], out_edges[2][2]} <= raster_rs[raster_mem_rsp_tag];
                    raster_rs_valid[raster_mem_rsp_tag] <= 0;
                    raster_rs_empty[raster_mem_rsp_tag] <= 1;
                end
            end

        end
    end

    // Calculate extents
    for (genvar i = 0; i < 3; ++i) begin
        VX_raster_extents #(
            .RASTER_TILE_SIZE(RASTER_TILE_SIZE)
        ) extent_calc (
            .edges(out_edges[i]),
            .extents(out_extents[i])
        );
    end


    // Priority encoder to select a free index in the RS
    VX_priority_encoder #( 
        .N(RASTER_RS_SIZE)
    ) raster_empty_rs (
        .data_in(raster_rs_empty),  
        `UNUSED_PIN (onehot),
        .index(raster_rs_empty_index),
        .valid_out(valid_rs_empty_index)
    );

    // Priority encoder to select the valid entry from RS to dispatch
    VX_priority_encoder #( 
        .N(RASTER_RS_SIZE)
    ) raster_request_rs (
        .data_in(raster_rs_valid),  
        `UNUSED_PIN (onehot),
        .index(raster_rs_index),
        .valid_out(valid_rs_index)
    );

    VX_priority_encoder #(
        .N(RASTER_SLICE_NUM)
    ) raster_ready_select (
        .data_in(raster_slice_ready),
        `UNUSED_PIN (onehot),
        .index(out_slice_index),
        .valid_out(valid_raster_index)
    );

    // Memory streamer
    VX_mem_streamer #(
        .NUM_REQS(9), // 3 edges and 3 coeffs in each edge
        .ADDRW(`RASTER_DCR_DATA_BITS),
        .DATAW(`RASTER_PRIMITIVE_DATA_BITS),
        .TAGW(RASTER_RS_INDEX_BITS)
    ) raster_mem_streamer (
        .clk(clk),
        .reset(reset),

        .req_valid(mem_req_valid),
        .req_rw(0),
        .req_mask(9'b1),
        `UNUSED_PIN (req_byteen),   /// TODO: USE THIS PIN
        .req_addr(mem_req_addr),
        `UNUSED_PIN (req_data),
        .req_tag(raster_rs_empty_index),
        .req_ready(mem_req_ready),
        
        // Output response
        .rsp_valid(mem_rsp_valid),
        `UNUSED_PIN (rsp_mask),
        .rsp_data(mem_rsp_data),
        .rsp_tag(raster_mem_rsp_tag),
        .rsp_ready(1),

        `UNUSED_PIN (mem_req_valid),
        `UNUSED_PIN (mem_req_rw),
        `UNUSED_PIN (mem_req_byteen),
        `UNUSED_PIN (mem_req_addr),
        `UNUSED_PIN (mem_req_data),
        `UNUSED_PIN (mem_req_tag),
        `UNUSED_PIN (mem_req_ready),
        `UNUSED_PIN (mem_rsp_valid),
        `UNUSED_PIN (mem_rsp_mask),
        `UNUSED_PIN (mem_rsp_data),
        `UNUSED_PIN (mem_rsp_tag),
        `UNUSED_PIN (mem_rsp_ready)
    );

endmodule
