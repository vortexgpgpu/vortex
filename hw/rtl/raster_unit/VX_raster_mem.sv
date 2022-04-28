`include "VX_raster_define.vh"

// Memory interface for the rasterization unit.
// Performs the following:
//  1. Break the request in tile and primitive fetch requests
//  2. Forms an FSM to keep a track of the return value types
//  3. Sotres the data in a reservation station
//  4. Dispatches the packets from the reservation station to the rasterslice
//      depending on the ready signal from the slices

module VX_raster_mem #(  
    parameter RASTER_SLICE_NUM  = 4,
    parameter RASTER_TILE_SIZE  = 16,
    parameter RASTER_RS_SIZE    = 8, // Reservation station size
    parameter RASTER_SLICE_BITS = `LOG2UP(RASTER_SLICE_NUM)
) (
    // Standard inputs
    input logic         clk,
    input logic         reset,

    // To indicate valid inputs provided
    input logic         input_valid,
    // Memory information
    input logic         [`RASTER_DCR_DATA_BITS-1:0]         num_tiles,
    input logic         [`RASTER_DCR_DATA_BITS-1:0]         tbuf_baseaddr,
    input logic         [`RASTER_DCR_DATA_BITS-1:0]         pbuf_baseaddr,
    input logic         [`RASTER_DCR_DATA_BITS-1:0]         pbuf_stride,

    // Raster slice interactions
    input logic         [RASTER_SLICE_NUM-1:0]              raster_slice_ready,
    output logic        [`RASTER_DIM_BITS-1:0]              out_x_loc, out_y_loc,
    output logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]   out_edges[2:0][2:0],
    output logic        [`RASTER_PRIMITIVE_DATA_BITS-1:0]   out_pid,
    output logic        [RASTER_SLICE_BITS-1:0]             out_slice_index,

    // Status signals
    output logic        ready, out_valid,

    // Memory interface
    VX_cache_req_if.master cache_req_if,
    VX_cache_rsp_if.slave  cache_rsp_if
);
    localparam MUL_LATENCY       = 3;
    localparam NUM_REQS          = `RASTER_MEM_REQS;
    localparam RASTER_TILE_BITS  = $clog2(RASTER_TILE_SIZE);

    // Bit tag identifier for type of memory request
    // Request #1: Tile and primitive id fetch => bit = 01
    // Request #2: Primitive id fetch => bit = 10
    // Request #3: Primitive data fetch => bit = 11
    localparam TILE_FETCH           = 2'b01;
    localparam PRIM_ID_FETCH        = 2'b10;
    localparam PRIM_DATA_FETCH      = 2'b11;
    localparam TILE_FETCH_MASK      = 9'(2'b11);
    localparam PRIM_ID_FETCH_MASK   = 9'(1'b1);
    localparam PRIM_DATA_FETCH_MASK = {9{1'b1}};

    // Temp storage to cycle through all primitives and tiles
    logic [`RASTER_DCR_DATA_BITS-1:0]   temp_tbuf_addr, temp_pbuf_addr,
                                        temp_num_prims, temp_num_tiles,
                                        temp_pbuf_stride;
    logic [`RASTER_DIM_BITS-1:0]        temp_x_loc, temp_y_loc;

    // Holds x_loc, y_loc, edge_func_val, edges, pid -> extents are calculated on the fly
    localparam RASTER_RS_DATA_WIDTH = 2*`RASTER_DIM_BITS + 3*3*`RASTER_PRIMITIVE_DATA_BITS + `RASTER_PRIMITIVE_DATA_BITS;
    localparam RASTER_RS_INDEX_BITS = `LOG2UP(RASTER_RS_SIZE);

    // Reservation station
    logic [RASTER_RS_DATA_WIDTH-1:0]    raster_rs[RASTER_RS_SIZE-1:0];
    logic [RASTER_RS_SIZE-1:0]          raster_rs_valid;
    logic [RASTER_RS_SIZE-1:0]          raster_rs_empty;

    logic [RASTER_RS_INDEX_BITS-1:0]    raster_rs_empty_index, raster_rs_index;

    // Status signals
    logic valid_raster_index, valid_rs_index, valid_rs_empty_index;
    logic fetch_fsm_complete;

    // Memory interactions
    logic mem_req_valid, mem_req_ready, mem_rsp_valid, temp_mem_rsp_valid;
    logic [8:0][`RASTER_DCR_DATA_BITS-1:0]       mem_req_addr;
    logic [8:0][`RASTER_PRIMITIVE_DATA_BITS-1:0] mem_rsp_data;
    logic [8:0][`RASTER_PRIMITIVE_DATA_BITS-1:0] prim_mem_rsp_data;
    logic [8:0][`RASTER_PRIMITIVE_DATA_BITS-1:0] temp_mem_rsp_data;

    localparam TAG_MAX_BIT_INDEX = 2;
    logic [TAG_MAX_BIT_INDEX-1:0]                mem_rsp_tag, temp_mem_rsp_tag; // size increased by 1 bit to account for the mem_tag_type
    logic [1:0]                                  mem_tag_type;
    logic [8:0]                                  mem_req_mask;
    logic [`RASTER_PRIMITIVE_DATA_BITS-1:0]      pid;

    // Stall when unable to fire the mem_req
    logic stall;

    // Stall signal
    //  -> assert when any entry in the RS is empty
    assign ready = |raster_rs_empty & mem_req_ready & fetch_fsm_complete;

    // Check for fsm completion
    assign fetch_fsm_complete = temp_num_tiles == 0;

    always @(posedge clk) begin
        // Setting default values:
        mem_req_valid       <= 0;
        // Reset condition
        if (reset) begin
            mem_req_valid   <= 0;
            out_valid       <= 0;
            temp_tbuf_addr  <= 0;
            temp_pbuf_addr  <= 0;
            temp_num_prims  <= 0;
            temp_num_tiles  <= 0;
            pid <= 0;
            for (int i = 0; i < RASTER_RS_SIZE; ++i) begin
                raster_rs_valid[i] <= 0;
                raster_rs_empty[i] <= 1;
            end
        end
        // Check for stall condition
        else if (stall && mem_req_valid) begin
            // re-assert the mem_req and don't change any other details
            mem_req_valid <= 1;
        end
        else begin
            // On new input -> set the temp state values
            if (input_valid && valid_rs_empty_index && fetch_fsm_complete == 1) begin
                temp_pbuf_addr   <= pbuf_baseaddr;
                temp_num_tiles   <= num_tiles;
                temp_pbuf_stride <= pbuf_stride;
                // Launch memory request to get the tbuf and pbuf
                mem_tag_type     <= TILE_FETCH;
                mem_req_mask     <= TILE_FETCH_MASK;
                mem_req_valid    <= 1;
                // Fetch the first primitive as well
                mem_req_addr[0]  <= tbuf_baseaddr;
                mem_req_addr[1]  <= tbuf_baseaddr + 4;
                // To indicate the address for next fetch
                temp_tbuf_addr   <= tbuf_baseaddr + 4 + 4;

            end
            // If it gets a valid memory response
            else if (mem_rsp_valid && fetch_fsm_complete == 0) begin
                // If not generate the tiles and primitives
                // Check the reponse tag type
                if (mem_rsp_tag[1:0] == TILE_FETCH) begin
                    // returned value is tile data
                    temp_x_loc      <= `RASTER_DIM_BITS'((mem_rsp_data[0] & {{16{1'b0}}, {16{1'b1}}}) << RASTER_TILE_BITS);
                    temp_y_loc      <= `RASTER_DIM_BITS'((mem_rsp_data[0] >> (16)) << RASTER_TILE_BITS);
                    temp_num_prims  <= mem_rsp_data[1];
                    // Launch the pid fetch
                    mem_tag_type    <= PRIM_ID_FETCH;
                    mem_req_valid   <= 1;
                    mem_req_mask    <= PRIM_ID_FETCH_MASK;
                    // Fetch the primitive index
                    mem_req_addr[0] <= temp_tbuf_addr;
                    temp_tbuf_addr  <= temp_tbuf_addr + 4;
                end
                else if (mem_rsp_tag[1:0] == PRIM_ID_FETCH) begin
                    // Launch next request based on pid
                    mem_tag_type <= PRIM_DATA_FETCH;
                    mem_req_valid <= 1;
                    mem_req_mask <= PRIM_DATA_FETCH_MASK;
                    pid <= mem_rsp_data[0];
                    for (int i = 0; i < 9; ++i) begin
                        mem_req_addr[i] <= prim_mem_rsp_data[i];
                    end
                end
                else if (mem_rsp_tag[1:0] == PRIM_DATA_FETCH) begin
                    // Insert data into RS
                    raster_rs[raster_rs_empty_index] <= {temp_x_loc, temp_y_loc,
                        mem_rsp_data[0], mem_rsp_data[1], mem_rsp_data[2],
                        mem_rsp_data[3], mem_rsp_data[4], mem_rsp_data[5],
                        mem_rsp_data[6], mem_rsp_data[7], mem_rsp_data[8],
                        pid
                    };
                    raster_rs_empty[raster_rs_empty_index] <= 0;
                    raster_rs_valid[raster_rs_empty_index] <= 1;

                    // Incrememnt the prim and tile count register
                    if (temp_num_prims - 1 == 0) begin
                        // => Last primitive fetched
                        temp_num_tiles <= temp_num_tiles - 1;
                        // Check if this was last tile
                        if (temp_num_tiles - 1 == 0) begin
                            // => this was last tile
                            temp_num_tiles <= temp_num_tiles - 1;
                        end
                        else begin
                            // Fetch the next tile
                            // Launch memory request to get the tbuf and pbuf
                            mem_tag_type    <= TILE_FETCH;
                            mem_req_valid   <= 1;
                            mem_req_mask    <= TILE_FETCH_MASK;
                            // Fetch the first primitive as well
                            mem_req_addr[0] <= temp_tbuf_addr;
                            mem_req_addr[1] <= temp_tbuf_addr + 4;
                            // To indicate the address for next fetch
                            temp_tbuf_addr  <= temp_tbuf_addr + 4 + 4;
                        end
                    end
                    else begin
                        temp_num_prims <= temp_num_prims - 1;
                        // Launch the request to get next primitive id // Launch the pid fetch
                        mem_tag_type    <= PRIM_ID_FETCH;
                        mem_req_valid   <= 1;
                        mem_req_mask    <= PRIM_ID_FETCH_MASK;
                        // Fetch the primitive index
                        mem_req_addr[0] <= temp_tbuf_addr;
                        temp_tbuf_addr  <= temp_tbuf_addr + 4;
                    end
                end
                else begin
                    `ASSERT(0, ("Incorrect tag returned"));
                    mem_req_valid <= 0;
                end
            end
        end
        // Launch any valid packet
        // When any raster slice is ready
        if (valid_raster_index && valid_rs_index && raster_slice_ready[out_slice_index] && out_valid == 0 &&
            raster_rs_valid[raster_rs_index]) begin
            {out_x_loc, out_y_loc,
            out_edges[0][0], out_edges[0][1], out_edges[0][2],
            out_edges[1][0], out_edges[1][1], out_edges[1][2],
            out_edges[2][0], out_edges[2][1], out_edges[2][2],
            out_pid} <= raster_rs[raster_rs_index];
            raster_rs_valid[raster_rs_index] <= 0;
            raster_rs_empty[raster_rs_index] <= 1;
            out_valid <= 1;
        end
        else begin
            out_valid <= 0;
        end
    end

    wire [`RASTER_PRIMITIVE_DATA_BITS-1:0] mul_result_tmp;

    VX_multiplier #(
        .WIDTHA  (`RASTER_PRIMITIVE_DATA_BITS),
        .WIDTHB  (`RASTER_PRIMITIVE_DATA_BITS),
        .WIDTHP  (`RASTER_PRIMITIVE_DATA_BITS),
        .SIGNED  (0),
        .LATENCY (MUL_LATENCY)
    ) multiplier (
        .clk    (clk),
        .enable (1'b1),
        .dataa  (temp_mem_rsp_data[0]),
        .datab  (temp_pbuf_stride),
        .result (mul_result_tmp)
    );
    for (genvar i = 0; i < 9; i++) begin
        assign prim_mem_rsp_data[i] = temp_pbuf_addr + mul_result_tmp + 4*i;
    end

    VX_shift_register #(
        .DATAW  (1 + TAG_MAX_BIT_INDEX + 9*`RASTER_PRIMITIVE_DATA_BITS),
        .DEPTH  (MUL_LATENCY),
        .RESETW (1 + TAG_MAX_BIT_INDEX)
    ) mul_shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({temp_mem_rsp_valid, temp_mem_rsp_tag,
            temp_mem_rsp_data[0], temp_mem_rsp_data[1], temp_mem_rsp_data[2],
            temp_mem_rsp_data[3], temp_mem_rsp_data[4], temp_mem_rsp_data[5],
            temp_mem_rsp_data[6], temp_mem_rsp_data[7], temp_mem_rsp_data[8]}),
        .data_out ({mem_rsp_valid, mem_rsp_tag,
            mem_rsp_data[0], mem_rsp_data[1], mem_rsp_data[2],
            mem_rsp_data[3], mem_rsp_data[4], mem_rsp_data[5],
            mem_rsp_data[6], mem_rsp_data[7], mem_rsp_data[8]})
    );

    // Priority encoder to select a free index in the RS
    VX_priority_encoder #( 
        .N          (RASTER_RS_SIZE)
    ) raster_empty_rs (
        .data_in    (raster_rs_empty),  
        `UNUSED_PIN (onehot),
        .index      (raster_rs_empty_index),
        .valid_out  (valid_rs_empty_index)
    );

    // Priority encoder to select the valid entry from RS to dispatch
    VX_priority_encoder #( 
        .N          (RASTER_RS_SIZE)
    ) raster_request_rs (
        .data_in    (raster_rs_valid),  
        `UNUSED_PIN (onehot),
        .index      (raster_rs_index),
        .valid_out  (valid_rs_index)
    );

    VX_priority_encoder #(
        .N          (RASTER_SLICE_NUM)
    ) raster_ready_select (
        .data_in    (raster_slice_ready),
        `UNUSED_PIN (onehot),
        .index      (out_slice_index),
        .valid_out  (valid_raster_index)
    );

    logic [8:0] [`RCACHE_ADDR_WIDTH-1:0] fire_mem_req_addr;
    for (genvar i = 0; i < 9; ++i) begin
        assign fire_mem_req_addr[i] = mem_req_addr[i][(32-`RCACHE_ADDR_WIDTH) +: `RCACHE_ADDR_WIDTH];
    end

    // Memory streamer
    wire mem_fire;
    assign mem_fire = mem_req_ready && mem_req_valid && |raster_rs_empty && !fetch_fsm_complete;
    assign stall = !mem_fire;
    VX_mem_streamer #(
        .NUM_REQS       (NUM_REQS), // 3 edges and 3 coeffs in each edge
        .NUM_BANKS      (`RCACHE_NUM_REQS),
        .ADDRW          (`RCACHE_ADDR_WIDTH),
        .DATAW          (`RASTER_PRIMITIVE_DATA_BITS),
        .QUEUE_SIZE     (`RASTER_MEM_QUEUE_SIZE),
        .TAGW           (TAG_MAX_BIT_INDEX), // the top bit will denote type of request
        .OUT_REG        (1)
    ) raster_mem_streamer (
        .clk            (clk),
        .reset          (reset),

        .req_valid      (mem_fire), // NOTE: This should ensure stalls
        .req_rw         (1'b0),
        .req_mask       (mem_req_mask),
        `UNUSED_PIN     (req_byteen),
        .req_addr       (fire_mem_req_addr),
        `UNUSED_PIN     (req_data),
        .req_tag        (mem_tag_type), //tag type appended to tag
        .req_ready      (mem_req_ready),
        
        // Output response
        .rsp_valid      (temp_mem_rsp_valid),
        `UNUSED_PIN     (rsp_mask),
        .rsp_data       (temp_mem_rsp_data),
        .rsp_tag        (temp_mem_rsp_tag),
        .rsp_ready      (1'b1),

        .mem_req_valid  (cache_req_if.valid),
        .mem_req_rw     (cache_req_if.rw),
        .mem_req_byteen (cache_req_if.byteen),
        .mem_req_addr   (cache_req_if.addr),
        .mem_req_data   (cache_req_if.data),
        .mem_req_tag    (cache_req_if.tag),
        .mem_req_ready  (cache_req_if.ready),

        .mem_rsp_valid  (cache_rsp_if.valid),
        .mem_rsp_data   (cache_rsp_if.data),
        .mem_rsp_tag    (cache_rsp_if.tag),
        .mem_rsp_ready  (cache_rsp_if.ready)
    );

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (out_valid) begin
            dpi_trace(2, "%d: raster-mem-out: x=%0d, y=%0d, pid=%0d, edge1.x=%0d, edge1.y=%0d, edge1.z=%0d, edge2.x=%0d, edge2.y=%0d, edge2.z=%0d, edge3.x=%0d, edge3.y=%0d, edge3.z=%0d\n",
                $time, out_x_loc, out_y_loc, out_pid,
                out_edges[0][0], out_edges[0][1], out_edges[0][2],
                out_edges[1][0], out_edges[1][1], out_edges[1][2],
                out_edges[2][0], out_edges[2][1], out_edges[2][2]);
        end
        if (|cache_req_if.valid) begin
            dpi_trace(3, "%d: raster-mem-cache-req: valid=", $time);
            `TRACE_ARRAY1D(3, cache_req_if.valid, 9);
            dpi_trace(3, ", addr=");
            `TRACE_ARRAY1D(3, cache_req_if.addr, 9);
            dpi_trace(3, ", tag=");
            `TRACE_ARRAY1D(3, cache_req_if.tag, 9);
            dpi_trace(3, "\n");
        end
        if (|cache_rsp_if.valid) begin
            dpi_trace(3, "%d: raster-mem-cache-rsp: valid=", $time);
            `TRACE_ARRAY1D(3, cache_rsp_if.valid, 9);
            dpi_trace(3, ", data=");
            `TRACE_ARRAY1D(3, cache_rsp_if.data, 9);
            dpi_trace(3, ", tag=");
            `TRACE_ARRAY1D(3, cache_rsp_if.tag, 9);
            dpi_trace(3, "\n");
        end  
    end
`endif

endmodule
