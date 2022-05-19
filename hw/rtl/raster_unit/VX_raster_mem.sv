`include "VX_raster_define.vh"

// Memory interface for the rasterization unit.
// Performs the following:
//  1. Break the request in tile and primitive fetch requests
//  2. Form an FSM to keep a track of the return value types
//  3. Store primitive data in an elastic buffer

module VX_raster_mem #(  
    parameter TILE_LOGSIZE = 16,
    parameter QUEUE_SIZE   = 8
) (
    input wire clk,
    input wire reset,

    // Device configurations
    raster_dcrs_t                       dcrs,

    // Memory interface
    VX_cache_req_if.master              cache_req_if,
    VX_cache_rsp_if.slave               cache_rsp_if,

    // Inputs
    input wire                          start,
    output wire                         busy,

    // Outputs
    output wire                         valid_out,
    output wire [`RASTER_PID_BITS-1:0] pid_out,
    output wire [`RASTER_DIM_BITS-1:0]  x_loc_out,
    output wire [`RASTER_DIM_BITS-1:0]  y_loc_out,
    output wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges_out,    
    input wire                          ready_out
);
    `UNUSED_VAR (dcrs)

    localparam MUL_LATENCY = 3;
    localparam NUM_REQS    = `RASTER_MEM_REQS;
    localparam FSM_BITS    = 2;
    localparam TAG_WIDTH   = `RASTER_PID_BITS;

    localparam STATE_IDLE = 2'b00;
    localparam STATE_TILE = 2'b01;
    localparam STATE_PRIM = 2'b10;
    
    localparam TILE_FETCH_MASK  = 9'(2'b11);
    localparam PID_FETCH_MASK   = 9'(1'b01);
    localparam PDATA_FETCH_MASK = {9{1'b1}};
    
    // A primitive data contains (x_loc, y_loc, pid, edges)
    localparam PRIM_DATA_WIDTH = 2 * `RASTER_DIM_BITS + `RASTER_PID_BITS + 9 * `RASTER_DATA_BITS;

    // Storage to cycle through all primitives and tiles
    reg [`RASTER_DCR_DATA_BITS-1:0] curr_tbuf_addr;
    reg [`RASTER_PID_BITS-1:0]      curr_num_prims;
    reg [`RASTER_PID_BITS-1:0]      rem_num_prims;
    reg [`RASTER_TILE_BITS-1:0]     curr_num_tiles;
    reg [`RASTER_DIM_BITS-1:0]      curr_x_loc;
    reg [`RASTER_DIM_BITS-1:0]      curr_y_loc;

    // Output buffer
    wire buf_out_valid;
    wire buf_out_ready;

    // Memory request
    reg mem_req_valid;
    reg [NUM_REQS-1:0] mem_req_mask;
    reg [8:0][`RASTER_DCR_DATA_BITS-1:0] mem_req_addr;
    reg [TAG_WIDTH-1:0] mem_req_tag;
    wire mem_req_ready;
    
    // Memory response
    wire mem_rsp_valid;    
    reg [NUM_REQS-1:0] mem_rsp_mask;
    wire [8:0][`RASTER_DATA_BITS-1:0] mem_rsp_data;    
    wire [TAG_WIDTH-1:0] mem_rsp_tag;
    wire mem_rsp_ready;

    wire prim_addr_rsp_valid;
    wire [8:0][`RASTER_DATA_BITS-1:0] prim_mem_addr;
    wire [`RASTER_PID_BITS-1:0] prim_id;

    // Memory fetch FSM

    reg [FSM_BITS-1:0] state;

    wire fsm_req_fire = mem_req_valid && mem_req_ready;

    wire prim_data_rsp_valid = mem_rsp_valid 
                            && (state == STATE_PRIM) 
                            && mem_rsp_mask[1];

    wire prim_data_rsp_fire = prim_data_rsp_valid && mem_rsp_ready;
    
    always @(posedge clk) begin
        if (reset) begin
            state          <= STATE_IDLE; 
            mem_req_valid  <= 0;
            curr_tbuf_addr <= 0;
            curr_num_prims <= 0;
            rem_num_prims  <= 0; 
            curr_num_tiles <= 0;          
        end begin
            // deassert valid when request is sent
            if (fsm_req_fire) begin
                mem_req_valid <= 0; 
            end

            case (state)
            STATE_IDLE: begin
                if (start && (dcrs.tile_count != 0)) begin
                    state           <= STATE_TILE;         
                    mem_req_valid   <= 1;
                    curr_num_tiles  <= dcrs.tile_count;                    
                    mem_req_addr[0] <= dcrs.tbuf_addr;
                    mem_req_addr[1] <= dcrs.tbuf_addr + 4;
                    mem_req_mask    <= TILE_FETCH_MASK;
                    mem_req_tag     <= 'x;
                    curr_tbuf_addr  <= dcrs.tbuf_addr + 4 + 4;
                end
            end
            STATE_TILE: begin
                if (mem_rsp_valid) begin
                    // handle tile header response
                    state           <= STATE_PRIM;
                    curr_x_loc      <= `RASTER_DIM_BITS'(mem_rsp_data[0][0 +: 16] << TILE_LOGSIZE);
                    curr_y_loc      <= `RASTER_DIM_BITS'(mem_rsp_data[0][16 +: 16] << TILE_LOGSIZE);                    
                    // send next primitive address
                    mem_req_valid   <= 1;   
                    mem_req_addr[0] <= curr_tbuf_addr;                    
                    mem_req_mask    <= PID_FETCH_MASK;                    
                    mem_req_tag     <= 'x;
                    curr_tbuf_addr  <= curr_tbuf_addr + 4;
                    curr_num_prims  <= mem_rsp_data[1][`RASTER_PID_BITS-1:0];
                    rem_num_prims   <= mem_rsp_data[1][`RASTER_PID_BITS-1:0];
                end
            end
            STATE_PRIM: begin
                if (prim_addr_rsp_valid) begin
                    // handle primitive address response  
                    mem_req_valid <= 1;
                    mem_req_addr  <= prim_mem_addr;
                    mem_req_mask  <= PDATA_FETCH_MASK;
                    mem_req_tag   <= prim_id;                        
                end else
                if (prim_data_rsp_fire) begin
                    // handle primitive data response
                    if (rem_num_prims == 1) begin
                        if (curr_num_tiles != 1) begin
                            // Fetch the next tile
                            state           <= STATE_TILE;
                            mem_req_valid   <= 1;
                            mem_req_addr[0] <= curr_tbuf_addr;
                            mem_req_addr[1] <= curr_tbuf_addr + 4;
                            mem_req_mask    <= TILE_FETCH_MASK;
                            mem_req_tag     <= 'x;
                            curr_tbuf_addr  <= curr_tbuf_addr + 4 + 4;
                            curr_num_tiles  <= curr_num_tiles - `RASTER_TILE_BITS'(1);
                        end else begin
                            // done, return to idle
                            state <= STATE_IDLE;
                        end
                    end
                    rem_num_prims <= rem_num_prims - `RASTER_PID_BITS'(1);
                end else                
                if (fsm_req_fire) begin
                    // send next primitive address
                    if (curr_num_prims != 1) begin        
                        mem_req_valid   <= 1;                
                        mem_req_addr[0] <= curr_tbuf_addr;                   
                        mem_req_mask    <= PID_FETCH_MASK;
                        mem_req_tag     <= 'x;
                        curr_tbuf_addr  <= curr_tbuf_addr + 4;
                        curr_num_prims  <= curr_num_prims - `RASTER_PID_BITS'(1);
                    end
                end
            end
            default:;
            endcase
        end
    end

    // Memory streamer

    // stall the memory response only if edge data cannot be taken
    assign mem_rsp_ready = (~prim_data_rsp_valid || buf_out_ready)
                        && ~prim_addr_rsp_valid;

    wire [8:0][`RCACHE_ADDR_WIDTH-1:0] mem_req_addr_w;
    for (genvar i = 0; i < 9; ++i) begin
        assign mem_req_addr_w[i] = mem_req_addr[i][(32 - `RCACHE_ADDR_WIDTH) +: `RCACHE_ADDR_WIDTH];
    end

    VX_mem_streamer #(
        .NUM_REQS   (NUM_REQS), 
        .NUM_BANKS  (`RCACHE_NUM_REQS),
        .ADDRW      (`RCACHE_ADDR_WIDTH),
        .DATAW      (`RASTER_DATA_BITS),
        .QUEUE_SIZE (`RASTER_MEM_PENDING_SIZE),
        .TAGW       (TAG_WIDTH),
        .OUT_REG    (1)
    ) mem_streamer (
        .clk            (clk),
        .reset          (reset),

        // Input request
        .req_valid      (mem_req_valid),
        .req_rw         (1'b0),
        .req_mask       (mem_req_mask),
        `UNUSED_PIN     (req_byteen),
        .req_addr       (mem_req_addr_w),
        `UNUSED_PIN     (req_data),
        .req_tag        (mem_req_tag),
        .req_ready      (mem_req_ready),
        
        // Output response
        .rsp_valid      (mem_rsp_valid),
        .rsp_mask       (mem_rsp_mask),
        .rsp_data       (mem_rsp_data),
        .rsp_tag        (mem_rsp_tag),
        .rsp_ready      (mem_rsp_ready),        

        // Memory request
        .mem_req_valid  (cache_req_if.valid),
        .mem_req_rw     (cache_req_if.rw),
        .mem_req_byteen (cache_req_if.byteen),
        .mem_req_addr   (cache_req_if.addr),
        .mem_req_data   (cache_req_if.data),
        .mem_req_tag    (cache_req_if.tag),
        .mem_req_ready  (cache_req_if.ready),

        // Memory response
        .mem_rsp_valid  (cache_rsp_if.valid),
        .mem_rsp_data   (cache_rsp_if.data),
        .mem_rsp_tag    (cache_rsp_if.tag),
        .mem_rsp_ready  (cache_rsp_if.ready)
    );

    wire [`RASTER_DATA_BITS-1:0] prim_mem_offset;

    VX_multiplier #(
        .WIDTHA  (`RASTER_DATA_BITS),
        .WIDTHB  (`RASTER_STRIDE_BITS),
        .WIDTHP  (`RASTER_DATA_BITS),
        .SIGNED  (0),
        .LATENCY (MUL_LATENCY)
    ) multiplier (
        .clk    (clk),
        .enable (1'b1),
        .dataa  (mem_rsp_data[0]),
        .datab  (dcrs.pbuf_stride),
        .result (prim_mem_offset)
    );

    for (genvar i = 0; i < 9; ++i) begin
        assign prim_mem_addr[i] = dcrs.pbuf_addr + prim_mem_offset + 4 * i;
    end

    // onlt delay primitive addresses for multiplication (mask = 1)
    wire mem_rsp_valid_p = mem_rsp_valid && ~mem_rsp_mask[1];

    VX_shift_register #(
        .DATAW  (1 + `RASTER_PID_BITS),
        .DEPTH  (MUL_LATENCY),
        .RESETW (1)
    ) mul_shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({mem_rsp_valid_p,     mem_rsp_data[0][`RASTER_PID_BITS-1:0]}),
        .data_out ({prim_addr_rsp_valid, prim_id})
    );   

    // Output buffer 

    assign buf_out_valid = prim_data_rsp_valid
                        && ~prim_addr_rsp_valid;
                      
    `UNUSED_VAR (mem_rsp_mask)

    VX_elastic_buffer #(
        .DATAW   (PRIM_DATA_WIDTH), 
        .SIZE    (QUEUE_SIZE),
        .OUT_REG (QUEUE_SIZE > 2)
    ) buf_out (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (buf_out_valid),
        .ready_in   (buf_out_ready),
        .data_in    ({curr_x_loc, curr_y_loc, mem_rsp_tag, mem_rsp_data}),                
        .data_out   ({x_loc_out,  y_loc_out,  pid_out,     edges_out}),
        .valid_out  (valid_out),
        .ready_out  (ready_out)
    );

    // busy ?
    assign busy = (state != STATE_IDLE);

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (valid_out && ready_out) begin
            dpi_trace(2, "%d: raster-mem-out: x=%0d, y=%0d, pid=%0d, edge={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}\n",
                $time, x_loc_out, y_loc_out, pid_out,
                edges_out[0][0], edges_out[0][1], edges_out[0][2],
                edges_out[1][0], edges_out[1][1], edges_out[1][2],
                edges_out[2][0], edges_out[2][1], edges_out[2][2]);
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
