`include "VX_raster_define.vh"

// Rasterizer Memory Unit
// Functionality:
//  1. Send memory request to fetch tile header.
//  2. Send memory request to fetch all primitives overlapping the tile.
//  3. Return the primitives with associated tile.

module VX_raster_mem #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX  = 0,
    parameter NUM_INSTANCES = 1, 
    parameter TILE_LOGSIZE  = 5,
    parameter QUEUE_SIZE    = 8
) (
    input wire clk,
    input wire reset,

    // Device configurations
    raster_dcrs_t                       dcrs,

    // Memory interface
    VX_cache_bus_if.master              cache_bus_if,

    // Inputs
    input wire                          start,
    output wire                         busy,

    // Outputs
    output wire                         valid_out,
    output wire [`RASTER_PID_BITS-1:0]  pid_out,
    output wire [`RASTER_DIM_BITS-1:0]  xloc_out,
    output wire [`RASTER_DIM_BITS-1:0]  yloc_out,
    output wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges_out,    
    input wire                          ready_out
);
    `UNUSED_VAR (dcrs)

    localparam LOG2_NUM_INSTANCES = `CLOG2(NUM_INSTANCES);

    localparam NUM_REQS         = RASTER_MEM_REQS;
    localparam FSM_BITS         = 2;
    localparam FETCH_FLAG_BITS  = 2;
    localparam TAG_WIDTH        = `RASTER_PID_BITS + FETCH_FLAG_BITS;

    localparam STATE_IDLE       = 2'b00;
    localparam STATE_TILE       = 2'b01;
    localparam STATE_PRIM       = 2'b10;
    
    localparam FETCH_FLAG_TILE  = 2'b00;
    localparam FETCH_FLAG_PID   = 2'b01;
    localparam FETCH_FLAG_PDATA = 2'b10;

    localparam TILE_HEADER_SIZEW = 8 / 4;
    
    // A primitive data contains (xloc, yloc, pid, edges)
    localparam PRIM_DATA_WIDTH = 2 * `RASTER_DIM_BITS + 9 * `RASTER_DATA_BITS + `RASTER_PID_BITS;

    // Storage to cycle through all primitives and tiles
    reg [RCACHE_ADDR_WIDTH-1:0] next_tbuf_addr;
    reg [RCACHE_ADDR_WIDTH-1:0] curr_pbuf_addr;
    reg [`RASTER_PID_BITS-1:0]  curr_pid_reqs;
    reg [`RASTER_PID_BITS-1:0]  curr_pid_rsps;
    reg [`RASTER_TILE_BITS-1:0] curr_num_tiles;
    reg [`RASTER_DIM_BITS-1:0]  curr_xloc;
    reg [`RASTER_DIM_BITS-1:0]  curr_yloc;

    // Output buffer
    wire buf_in_valid;
    wire buf_in_ready;

    // Memory request
    reg mem_req_valid, mem_req_valid_qual;
    reg [NUM_REQS-1:0] mem_req_mask;
    reg [8:0][RCACHE_ADDR_WIDTH-1:0] mem_req_addr;
    reg [TAG_WIDTH-1:0] mem_req_tag;
    wire mem_req_ready;
    
    // Memory response
    wire mem_rsp_valid;
    wire [8:0][`RASTER_DATA_BITS-1:0] mem_rsp_data;    
    wire [TAG_WIDTH-1:0] mem_rsp_tag;
    wire mem_rsp_ready;
    
     // Primitive info
    wire [RCACHE_ADDR_WIDTH-1:0] pids_addr;
    wire prim_id_rsp_valid;
    wire prim_data_rsp_valid;
    wire prim_addr_rsp_valid;
    wire prim_addr_rsp_ready;
    wire [8:0][RCACHE_ADDR_WIDTH-1:0] prim_mem_addr;
    wire [`RASTER_PID_BITS-1:0] primitive_id;

    // Memory fetch FSM

    reg [FSM_BITS-1:0] state;
    
    wire is_prim_id_req   = (mem_req_tag[FETCH_FLAG_BITS-1:0] == FETCH_FLAG_PID);
    wire is_prim_id_rsp   = (mem_rsp_tag[FETCH_FLAG_BITS-1:0] == FETCH_FLAG_PID);

    wire is_prim_data_req = (mem_req_tag[FETCH_FLAG_BITS-1:0] == FETCH_FLAG_PDATA);
    wire is_prim_data_rsp = (mem_rsp_tag[FETCH_FLAG_BITS-1:0] == FETCH_FLAG_PDATA);

    wire mem_req_fire = mem_req_valid_qual && mem_req_ready;

    wire prim_addr_rsp_fire = prim_addr_rsp_valid && prim_addr_rsp_ready;

    wire prim_data_rsp_fire = prim_data_rsp_valid && mem_rsp_ready;

    // tile header info
    wire [15:0] th_tile_pos_x  = mem_rsp_data[0][0  +: 16];
    wire [15:0] th_tile_pos_y  = mem_rsp_data[0][16 +: 16];
    wire [15:0] th_pids_offset = mem_rsp_data[1][0  +: 16];
    wire [15:0] th_pids_count  = mem_rsp_data[1][16 +: 16];

    // calculate tile start info
    wire [`RASTER_TILE_BITS-1:0] start_tile_count = (dcrs.tile_count + `RASTER_TILE_BITS'(NUM_INSTANCES - 1 - INSTANCE_IDX)) >> LOG2_NUM_INSTANCES;
    wire [RCACHE_ADDR_WIDTH-1:0] start_tbuf_addr = RCACHE_ADDR_WIDTH'({dcrs.tbuf_addr, 4'b0}) + RCACHE_ADDR_WIDTH'(INSTANCE_IDX * TILE_HEADER_SIZEW);

    // calculate address of primitive ids
    assign pids_addr = (mem_req_addr[1] + 1) + RCACHE_ADDR_WIDTH'(th_pids_offset);
    
    // scheduler FSM
    always @(posedge clk) begin
        if (reset) begin
            state <= STATE_IDLE; 
            mem_req_valid <= 0;    
        end else begin
            // deassert memory request when fired
            if (mem_req_fire) begin
                mem_req_valid <= 0; 
            end

            case (state)
            STATE_IDLE: begin
                // fetch the next tile header
                if (start && (start_tile_count != 0)) begin
                    state <= STATE_TILE;
                end
                mem_req_valid   <= 1;
                mem_req_addr[0] <= start_tbuf_addr;
                mem_req_addr[1] <= start_tbuf_addr + 1;
                mem_req_mask    <= 9'b11;
                mem_req_tag     <= TAG_WIDTH'(FETCH_FLAG_TILE);
                // update tile counters
                next_tbuf_addr  <= start_tbuf_addr + RCACHE_ADDR_WIDTH'(NUM_INSTANCES * TILE_HEADER_SIZEW);
                curr_num_tiles  <= start_tile_count;
            end
            STATE_TILE: begin
                if (mem_rsp_valid) begin
                    // handle tile header response
                    state           <= STATE_PRIM;
                    curr_xloc       <= `RASTER_DIM_BITS'(th_tile_pos_x << TILE_LOGSIZE);
                    curr_yloc       <= `RASTER_DIM_BITS'(th_tile_pos_y << TILE_LOGSIZE);                    
                    // fetch next primitive pid
                    mem_req_valid   <= 1;   
                    mem_req_addr[0] <= pids_addr;
                    mem_req_mask    <= 9'b1;                    
                    mem_req_tag     <= TAG_WIDTH'(FETCH_FLAG_PID);
                    // set primitive counters
                    curr_pbuf_addr  <= pids_addr;
                    curr_pid_reqs   <= `RASTER_PID_BITS'(th_pids_count);
                    curr_pid_rsps   <= `RASTER_PID_BITS'(th_pids_count);
                end
            end
            STATE_PRIM: begin
                // handle memory submissions
                if (mem_req_fire) begin
                    if (is_prim_id_req) begin
                        // update pid counters
                        curr_pbuf_addr <= curr_pbuf_addr + 1;
                        curr_pid_reqs  <= curr_pid_reqs - `RASTER_PID_BITS'(1);
                    end

                    if ((curr_pid_reqs > 1) 
                     || (curr_pid_reqs == 1 && ~is_prim_id_req)) begin
                        // fetch next primitive pid
                        mem_req_valid   <= 1;                        
                        mem_req_mask    <= 9'b1;
                        mem_req_addr[0] <= curr_pbuf_addr + (is_prim_id_req ? 1 : 0);
                        mem_req_tag     <= TAG_WIDTH'(FETCH_FLAG_PID);                        
                    end
                end

                // handle primitive address response  
                if (prim_addr_rsp_fire) begin                    
                    mem_req_valid <= 1;                    
                    mem_req_mask  <= 9'b111111111;
                    mem_req_addr  <= prim_mem_addr;
                    mem_req_tag   <= TAG_WIDTH'({primitive_id, FETCH_FLAG_PDATA});
                end 
                
                // handle primitive data response
                if (prim_data_rsp_fire) begin
                    if (curr_pid_rsps == 1) begin
                        if (curr_num_tiles == 1) begin
                            // done, return to idle
                            state <= STATE_IDLE;
                        end else begin
                            // fetch the next tile header
                            state           <= STATE_TILE;
                            mem_req_valid   <= 1;
                            mem_req_mask    <= 9'b11;
                            mem_req_addr[0] <= next_tbuf_addr;
                            mem_req_addr[1] <= next_tbuf_addr + 1;                            
                            mem_req_tag     <= TAG_WIDTH'(FETCH_FLAG_TILE);
                            next_tbuf_addr  <= next_tbuf_addr + RCACHE_ADDR_WIDTH'(NUM_INSTANCES * TILE_HEADER_SIZEW);
                        end
                        // update tile counter
                        curr_num_tiles <= curr_num_tiles - `RASTER_TILE_BITS'(1);
                    end
                    // update pid counter
                    curr_pid_rsps <= curr_pid_rsps - `RASTER_PID_BITS'(1);
                end
            end
            default:;
            endcase
        end
    end

    // Memory streamer

    // ensure that we have space in the output buffer to prevent memory deadlock
    wire pending_output_full;
    VX_pending_size #( 
        .SIZE (QUEUE_SIZE-1)
    ) pending_reads (
        .clk   (clk),
        .reset (reset),
        .incr  (mem_req_fire && is_prim_id_req),
        .decr  (valid_out && ready_out),
        .full  (pending_output_full),
        `UNUSED_PIN (size),
        `UNUSED_PIN (empty)
    );    
    assign mem_req_valid_qual = mem_req_valid && (~is_prim_id_req || ~pending_output_full);

    // the memory response is for primitive id
    assign prim_id_rsp_valid = mem_rsp_valid && is_prim_id_rsp;

    // the memory response is for primitive data
    assign prim_data_rsp_valid = mem_rsp_valid && is_prim_data_rsp;

    // stall primitive address handling if primitive data fetch stalls
    wire prim_data_req_stall = mem_req_valid && is_prim_data_req && ~mem_req_ready;
    assign prim_addr_rsp_ready = ~prim_data_req_stall || ~prim_addr_rsp_valid;

    // Push primitive data into output buffer
    assign buf_in_valid = prim_data_rsp_valid;

    // stall the memory response
    assign mem_rsp_ready = (~prim_id_rsp_valid || prim_addr_rsp_ready) 
                        && (~prim_data_rsp_valid || buf_in_ready);

    wire [8:0][RCACHE_WORD_SIZE-1:0] mem_req_byteen;
    for (genvar i = 0; i < 9; ++i) begin
        assign mem_req_byteen[i] = {RCACHE_WORD_SIZE{1'b1}};
    end

    // schedule memory request
    VX_mem_scheduler #(
        .INSTANCE_ID  ($sformatf("%s-memsched", INSTANCE_ID)),
        .NUM_REQS     (NUM_REQS), 
        .NUM_BANKS    (RCACHE_NUM_REQS),
        .ADDR_WIDTH   (RCACHE_ADDR_WIDTH),
        .DATA_WIDTH   (`RASTER_DATA_BITS),
        .QUEUE_SIZE   (`RASTER_MEM_QUEUE_SIZE),
        .TAG_WIDTH    (TAG_WIDTH),
        .CORE_OUT_REG (2),
        .MEM_OUT_REG  (3)
    ) mem_scheduler (
        .clk            (clk),
        .reset          (reset),

        // Input request
        .req_valid      (mem_req_valid_qual),
        .req_rw         (1'b0),
        .req_mask       (mem_req_mask),
        .req_byteen     (mem_req_byteen),
        .req_addr       (mem_req_addr),
        `UNUSED_PIN     (req_data),
        .req_tag        (mem_req_tag),
        `UNUSED_PIN     (req_empty),
        .req_ready      (mem_req_ready),
        `UNUSED_PIN     (write_notify),
        
        // Output response
        .rsp_valid      (mem_rsp_valid),
        `UNUSED_PIN     (rsp_mask),
        .rsp_data       (mem_rsp_data),
        .rsp_tag        (mem_rsp_tag),
        `UNUSED_PIN     (rsp_eop),
        .rsp_ready      (mem_rsp_ready),

        // Memory request
        .mem_req_valid  (cache_bus_if.req_valid),
        .mem_req_rw     (cache_bus_if.req_rw),
        .mem_req_byteen (cache_bus_if.req_byteen),
        .mem_req_addr   (cache_bus_if.req_addr),
        .mem_req_data   (cache_bus_if.req_data),
        .mem_req_tag    (cache_bus_if.req_tag),
        .mem_req_ready  (cache_bus_if.req_ready),

        // Memory response
        .mem_rsp_valid  (cache_bus_if.rsp_valid),
        .mem_rsp_data   (cache_bus_if.rsp_data),
        .mem_rsp_tag    (cache_bus_if.rsp_tag),
        .mem_rsp_ready  (cache_bus_if.rsp_ready)
    );

    wire [RCACHE_ADDR_WIDTH-1:0] prim_mem_offset;

    VX_multiplier #(
        .A_WIDTH (`RASTER_DATA_BITS),
        .B_WIDTH (`RASTER_STRIDE_BITS),
        .R_WIDTH (RCACHE_ADDR_WIDTH),
        .LATENCY (`LATENCY_IMUL)
    ) multiplier (
        .clk    (clk),
        .enable (prim_addr_rsp_ready),
        .dataa  (mem_rsp_data[0]),
        .datab  (dcrs.pbuf_stride),
        .result (prim_mem_offset)
    );

    for (genvar i = 0; i < 9; ++i) begin
        wire [RCACHE_ADDR_WIDTH-1:0] offset = prim_mem_offset + RCACHE_ADDR_WIDTH'(1 * i);
        assign prim_mem_addr[i] = RCACHE_ADDR_WIDTH'({dcrs.pbuf_addr, 4'b0}) + offset;
    end

    VX_shift_register #(
        .DATAW  (1 + `RASTER_PID_BITS),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) mul_shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (prim_addr_rsp_ready),
        .data_in  ({prim_id_rsp_valid,   mem_rsp_data[0][`RASTER_PID_BITS-1:0]}),
        .data_out ({prim_addr_rsp_valid, primitive_id})
    );   

    // Output buffer
    VX_elastic_buffer #(
        .DATAW   (PRIM_DATA_WIDTH), 
        .SIZE    (QUEUE_SIZE),
        .OUT_REG (1)
    ) buf_out (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (buf_in_valid),
        .ready_in   (buf_in_ready),
        .data_in    ({curr_xloc, curr_yloc, mem_rsp_data, mem_rsp_tag[FETCH_FLAG_BITS +: `RASTER_PID_BITS]}),                
        .data_out   ({xloc_out,  yloc_out,  edges_out,    pid_out}),
        .valid_out  (valid_out),
        .ready_out  (ready_out)
    );

    // busy ?
    assign busy = (state != STATE_IDLE);

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (valid_out && ready_out) begin
            `TRACE(2, ("%d: %s-mem-out: x=%0d, y=%0d, pid=%0d, edge={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}\n",
                $time, INSTANCE_ID, xloc_out, yloc_out, pid_out,
                edges_out[0][0], edges_out[0][1], edges_out[0][2],
                edges_out[1][0], edges_out[1][1], edges_out[1][2],
                edges_out[2][0], edges_out[2][1], edges_out[2][2]));
        end 
    end
`endif

endmodule
