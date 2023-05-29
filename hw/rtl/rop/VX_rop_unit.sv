`include "VX_rop_define.vh"

module VX_rop_unit #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 4
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_rop_perf_if.master   perf_rop_if,
`endif

    // Memory interface
    VX_cache_req_if.master  cache_req_if,
    VX_cache_rsp_if.slave   cache_rsp_if,

    // Inputs
    VX_dcr_write_if.slave   dcr_write_if,
    VX_rop_req_if.slave     rop_req_if
);
    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam MEM_TAG_WIDTH = UUID_WIDTH + NUM_LANES * (`ROP_DIM_BITS + `ROP_DIM_BITS + 32 + `ROP_DEPTH_BITS + 1);
    localparam DS_TAG_WIDTH = NUM_LANES * (`ROP_DIM_BITS + `ROP_DIM_BITS + 1 + 1 + 32);
    localparam BLEND_TAG_WIDTH  = NUM_LANES * (`ROP_DIM_BITS + `ROP_DIM_BITS + 1);

    // DCRs

    rop_dcrs_t rop_dcrs;

    VX_rop_dcr #( 
        .INSTANCE_ID (INSTANCE_ID)
    ) rop_dcr (
        .clk        (clk),
        .reset      (reset),
        .dcr_write_if(dcr_write_if),
        .rop_dcrs   (rop_dcrs)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire                                    mem_req_valid, mem_req_valid_r;
    wire [NUM_LANES-1:0]                    mem_req_ds_mask, mem_req_ds_mask_r;
    wire [NUM_LANES-1:0]                    mem_req_c_mask, mem_req_c_mask_r;
    wire                                    mem_req_rw, mem_req_rw_r;
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] mem_req_pos_x, mem_req_pos_x_r;
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] mem_req_pos_y, mem_req_pos_y_r;
    rgba_t [NUM_LANES-1:0]                  mem_req_color, mem_req_color_r;
    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] mem_req_depth, mem_req_depth_r;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] mem_req_stencil, mem_req_stencil_r;
    wire [NUM_LANES-1:0]                    mem_req_face, mem_req_face_r;
    wire [MEM_TAG_WIDTH-1:0]                mem_req_tag, mem_req_tag_r;
    wire                                    mem_req_ready, mem_req_ready_r;

    wire                                    mem_rsp_valid;
    wire [NUM_LANES-1:0]                    mem_rsp_mask;
    rgba_t [NUM_LANES-1:0]                  mem_rsp_color;
    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] mem_rsp_depth;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] mem_rsp_stencil;
    wire [MEM_TAG_WIDTH-1:0]                mem_rsp_tag;
    wire                                    mem_rsp_ready;
    wire                                    mem_write_notify;

    `RESET_RELAY (mem_reset, reset);

    VX_rop_mem #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (MEM_TAG_WIDTH)
    ) rop_mem (
        .clk            (clk),
        .reset          (mem_reset),

        .dcrs           (rop_dcrs),

        .cache_req_if   (cache_req_if),
        .cache_rsp_if   (cache_rsp_if),

        .req_valid      (mem_req_valid_r),
        .req_ds_mask    (mem_req_ds_mask_r),
        .req_c_mask     (mem_req_c_mask_r),
        .req_rw         (mem_req_rw_r),
        .req_pos_x      (mem_req_pos_x_r),
        .req_pos_y      (mem_req_pos_y_r),
        .req_color      (mem_req_color_r), 
        .req_depth      (mem_req_depth_r),
        .req_stencil    (mem_req_stencil_r),
        .req_face       (mem_req_face_r),
        .req_tag        (mem_req_tag_r),
        .req_ready      (mem_req_ready_r),
        .write_notify   (mem_write_notify),

        .rsp_valid      (mem_rsp_valid),
        .rsp_mask       (mem_rsp_mask),
        .rsp_color      (mem_rsp_color), 
        .rsp_depth      (mem_rsp_depth),
        .rsp_stencil    (mem_rsp_stencil),
        .rsp_tag        (mem_rsp_tag),
        .rsp_ready      (mem_rsp_ready)       
    );

    ///////////////////////////////////////////////////////////////////////////

    wire                    ds_valid_in;
    wire [DS_TAG_WIDTH-1:0] ds_tag_in;
    wire                    ds_ready_in;   
    wire                    ds_valid_out;
    wire [DS_TAG_WIDTH-1:0] ds_tag_out;
    wire                    ds_ready_out;

    wire [NUM_LANES-1:0]    ds_face;

    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]   ds_depth_ref;
    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]   ds_depth_val;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] ds_stencil_val;

    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]   ds_depth_out;      
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] ds_stencil_out;
    wire [NUM_LANES-1:0]                        ds_pass_out;

    `RESET_RELAY (ds_reset, reset);

    VX_rop_ds #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (DS_TAG_WIDTH)
    ) rop_ds (
        .clk            (clk),
        .reset          (ds_reset),

        .dcrs           (rop_dcrs),

        .valid_in       (ds_valid_in),      
        .tag_in         (ds_tag_in), 
        .ready_in       (ds_ready_in), 

        .valid_out      (ds_valid_out),
        .tag_out        (ds_tag_out),
        .ready_out      (ds_ready_out),

        .face           (ds_face),
        .depth_ref      (ds_depth_ref),
        .depth_val      (ds_depth_val),
        .stencil_val    (ds_stencil_val),    

        .depth_out      (ds_depth_out),        
        .stencil_out    (ds_stencil_out),
        .pass_out       (ds_pass_out)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire                    blend_valid_in;
    wire [BLEND_TAG_WIDTH-1:0] blend_tag_in;
    wire                    blend_ready_in;   
    wire                    blend_valid_out;
    wire [BLEND_TAG_WIDTH-1:0] blend_tag_out;
    wire                    blend_ready_out;

    rgba_t [NUM_LANES-1:0]  blend_src_color;
    rgba_t [NUM_LANES-1:0]  blend_dst_color;
    rgba_t [NUM_LANES-1:0]  blend_color_out;

    `RESET_RELAY (blend_reset, reset);

    VX_rop_blend #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (BLEND_TAG_WIDTH)
    ) rop_blend (
        .clk            (clk),
        .reset          (blend_reset),

        .dcrs           (rop_dcrs),

        .valid_in       (blend_valid_in),      
        .tag_in         (blend_tag_in),
        .ready_in       (blend_ready_in), 

        .valid_out      (blend_valid_out),
        .tag_out        (blend_tag_out),
        .ready_out      (blend_ready_out),
        
        .src_color      (blend_src_color),
        .dst_color      (blend_dst_color),
        .color_out      (blend_color_out)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire color_writeen = (rop_dcrs.cbuf_writemask != 0);

    wire depth_enable  = rop_dcrs.depth_enable;
    wire depth_writeen = rop_dcrs.depth_enable && (rop_dcrs.depth_writemask != 0);

    wire stencil_enable  = (| rop_dcrs.stencil_enable);
    wire stencil_writeen = (rop_dcrs.stencil_enable[0] && (rop_dcrs.stencil_writemask[0] != 0))
                        || (rop_dcrs.stencil_enable[1] && (rop_dcrs.stencil_writemask[1] != 0));

    wire ds_enable  = depth_enable || stencil_enable;
    wire ds_writeen = depth_writeen || stencil_writeen;

    wire blend_enable  = rop_dcrs.blend_enable;
    wire blend_writeen = rop_dcrs.blend_enable && color_writeen;

    wire ds_color_writeen = ds_writeen || (ds_enable && color_writeen);

    wire mem_readen = ds_color_writeen || blend_writeen;

    wire write_bypass = ~ds_enable && ~blend_enable && color_writeen;

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] mem_rsp_pos_x, mem_rsp_pos_y;
    wire [UUID_WIDTH-1:0] mem_rsp_uuid;
    `UNUSED_VAR (mem_rsp_uuid)

    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] ds_write_pos_x, ds_write_pos_y;
    wire [NUM_LANES-1:0] ds_write_face, ds_rsp_mask;
    rgba_t [NUM_LANES-1:0] ds_write_color;

    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] blend_write_pos_x, blend_write_pos_y;
    wire [NUM_LANES-1:0] blend_rsp_mask;

    wire pending_reads_full;
    
    assign mem_req_tag = {rop_req_if.uuid, rop_req_if.pos_x, rop_req_if.pos_y, rop_req_if.color, rop_req_if.depth, rop_req_if.face};
    assign {mem_rsp_uuid, mem_rsp_pos_x, mem_rsp_pos_y, blend_src_color, ds_depth_ref, ds_face} = mem_rsp_tag;

    assign ds_tag_in = {mem_rsp_pos_x, mem_rsp_pos_y, mem_rsp_mask, ds_face, blend_src_color};
    assign {ds_write_pos_x, ds_write_pos_y, ds_rsp_mask, ds_write_face, ds_write_color} = ds_tag_out;

    assign blend_tag_in = {mem_rsp_pos_x, mem_rsp_pos_y, mem_rsp_mask};
    assign {blend_write_pos_x, blend_write_pos_y, blend_rsp_mask} = blend_tag_out;

    wire color_write = write_bypass && rop_req_if.valid;

    wire ds_blend_read = mem_readen && rop_req_if.valid && ~pending_reads_full;

    wire ds_blend_write = (ds_color_writeen && blend_writeen) ? (ds_valid_out && blend_valid_out) :
                            (ds_color_writeen ? ds_valid_out :
                                (blend_writeen ? blend_valid_out :
                                    1'b0));

    wire [NUM_LANES-1:0] ds_read_mask, ds_write_mask;
    wire [NUM_LANES-1:0] blend_read_mask, blend_write_mask;
    wire [NUM_LANES-1:0] color_bypass_mask, ds_color_write_mask;

    for (genvar i = 0;  i < NUM_LANES; ++i) begin      
        assign ds_read_mask[i]        = rop_req_if.mask[i] && ds_enable;
        assign blend_read_mask[i]     = rop_req_if.mask[i] && blend_writeen;
        assign ds_write_mask[i]       = ds_rsp_mask[i] && (stencil_writeen || (depth_writeen && ds_pass_out[i]));
        assign blend_write_mask[i]    = blend_rsp_mask[i] && blend_writeen && (~ds_enable || ds_pass_out[i]);  
        assign color_bypass_mask[i]   = rop_req_if.mask[i] && color_writeen;
        assign ds_color_write_mask[i] = ds_rsp_mask[i] && ds_pass_out[i];
    end

    assign mem_req_valid    = ds_blend_write || ds_blend_read || color_write;
    assign mem_req_ds_mask  = ds_valid_out ? ds_write_mask : ds_read_mask;
    assign mem_req_c_mask   = write_bypass ? color_bypass_mask : (blend_valid_out ? blend_write_mask : (ds_valid_out ? ds_color_write_mask : blend_read_mask));
    assign mem_req_rw       = ds_blend_write || write_bypass;
    assign mem_req_face     = ds_write_face;
    assign mem_req_pos_x    = ds_valid_out ? ds_write_pos_x : (blend_valid_out ? blend_write_pos_x : rop_req_if.pos_x);
    assign mem_req_pos_y    = ds_valid_out ? ds_write_pos_y : (blend_valid_out ? blend_write_pos_y : rop_req_if.pos_y);
    assign mem_req_color    = blend_enable ? blend_color_out : (ds_enable ? ds_write_color : rop_req_if.color);
    assign mem_req_depth    = ds_depth_out;
    assign mem_req_stencil  = ds_stencil_out;
    
    assign ds_ready_out     = mem_req_ready && (~blend_writeen || blend_valid_out);
    assign blend_ready_out  = mem_req_ready && (~ds_color_writeen || ds_valid_out);
    assign rop_req_if.ready = mem_req_ready && ~ds_blend_write && ~pending_reads_full;

    assign ds_valid_in      = ds_enable && mem_rsp_valid && (~blend_enable || blend_ready_in);
    assign blend_valid_in   = blend_enable && mem_rsp_valid && (~ds_enable || ds_ready_in);
    assign blend_dst_color  = mem_rsp_color;    

    assign ds_depth_val     = mem_rsp_depth;
    assign ds_stencil_val   = mem_rsp_stencil;    
    assign mem_rsp_ready    = (ds_enable && blend_enable) ? (ds_ready_in && blend_ready_in) :
                                (ds_enable ? ds_ready_in :
                                    (blend_enable ? blend_ready_in :
                                        1'b0));

    wire mem_req_fire = mem_req_valid && mem_req_ready;

    wire write_req_canceled;

    // to prevent potential deadlocks, 
    // ensure the memory scheduler's queue doesn't fill up
    VX_pending_size #( 
        .SIZE (`ROP_MEM_QUEUE_SIZE)
    ) pending_reads (
        .clk   (clk),
        .reset (reset),
        .incr  (mem_req_fire && ~mem_req_rw && (ds_color_writeen || blend_writeen)),
        .decr  ((mem_write_notify || write_req_canceled) && (ds_color_writeen || blend_writeen)),
        .full  (pending_reads_full),
        `UNUSED_PIN (size),
        `UNUSED_PIN (empty)
    );
    
    wire mem_req_valid_unqual_r;

    VX_generic_buffer #(
        .DATAW	 (1 + NUM_LANES * (1 + 1 + 2 * `ROP_DIM_BITS + $bits(rgba_t) + `ROP_DEPTH_BITS + `ROP_STENCIL_BITS + 1) + MEM_TAG_WIDTH),
        .OUT_REG (1)
    ) mem_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_req_valid),
        .ready_in  (mem_req_ready),
        .data_in   ({mem_req_rw,   mem_req_ds_mask,   mem_req_c_mask,   mem_req_pos_x,   mem_req_pos_y,   mem_req_color,   mem_req_depth,   mem_req_stencil,   mem_req_face,   mem_req_tag}),
        .data_out  ({mem_req_rw_r, mem_req_ds_mask_r, mem_req_c_mask_r, mem_req_pos_x_r, mem_req_pos_y_r, mem_req_color_r, mem_req_depth_r, mem_req_stencil_r, mem_req_face_r, mem_req_tag_r}),
        .valid_out (mem_req_valid_unqual_r),
        .ready_out (mem_req_ready_r)
    );

    wire is_degenerate_req = (mem_req_ds_mask_r | mem_req_c_mask_r) == 0;

    assign mem_req_valid_r = mem_req_valid_unqual_r && ~is_degenerate_req;

    assign write_req_canceled = mem_req_valid_unqual_r && mem_req_rw_r && is_degenerate_req && mem_req_ready_r;

`ifdef PERF_ENABLE

    wire [$clog2(OCACHE_NUM_REQS+1)-1:0] perf_mem_rd_req_per_cycle;
    wire [$clog2(OCACHE_NUM_REQS+1)-1:0] perf_mem_wr_req_per_cycle;
    wire [$clog2(OCACHE_NUM_REQS+1)-1:0] perf_mem_rd_rsp_per_cycle;
    wire [$clog2(OCACHE_NUM_REQS+1)+1-1:0] perf_pending_reads_cycle;

    wire [OCACHE_NUM_REQS-1:0] perf_mem_rd_req_fire = cache_req_if.valid & ~cache_req_if.rw & cache_req_if.ready;
    wire [OCACHE_NUM_REQS-1:0] perf_mem_wr_req_fire = cache_req_if.valid & cache_req_if.rw & cache_req_if.ready;
    wire [OCACHE_NUM_REQS-1:0] perf_mem_rd_rsp_fire = cache_rsp_if.valid & cache_rsp_if.ready;

    `POP_COUNT(perf_mem_rd_req_per_cycle, perf_mem_rd_req_fire);    
    `POP_COUNT(perf_mem_wr_req_per_cycle, perf_mem_wr_req_fire);    
    `POP_COUNT(perf_mem_rd_rsp_per_cycle, perf_mem_rd_rsp_fire);

    reg [`PERF_CTR_BITS-1:0] perf_pending_reads;   
    assign perf_pending_reads_cycle = perf_mem_rd_req_per_cycle - perf_mem_rd_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_pending_reads <= '0;
        end else begin
            perf_pending_reads <= $signed(perf_pending_reads) + `PERF_CTR_BITS'($signed(perf_pending_reads_cycle));
        end
    end

    wire perf_stall_cycle = rop_req_if.valid & ~rop_req_if.ready;

    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_writes;
    reg [`PERF_CTR_BITS-1:0] perf_mem_latency;
    reg [`PERF_CTR_BITS-1:0] perf_stall_cycles;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_reads    <= '0;
            perf_mem_writes   <= '0;
            perf_mem_latency  <= '0;
            perf_stall_cycles <= '0;
        end else begin
            perf_mem_reads    <= perf_mem_reads    + `PERF_CTR_BITS'(perf_mem_rd_req_per_cycle);
            perf_mem_writes   <= perf_mem_writes   + `PERF_CTR_BITS'(perf_mem_wr_req_per_cycle);
            perf_mem_latency  <= perf_mem_latency  + `PERF_CTR_BITS'(perf_pending_reads);
            perf_stall_cycles <= perf_stall_cycles + `PERF_CTR_BITS'(perf_stall_cycle);
        end
    end

    assign perf_rop_if.mem_reads    = perf_mem_reads;
    assign perf_rop_if.mem_writes   = perf_mem_writes;
    assign perf_rop_if.mem_latency  = perf_mem_latency;
    assign perf_rop_if.stall_cycles = perf_stall_cycles;

`endif

endmodule

///////////////////////////////////////////////////////////////////////////////

module VX_rop_unit_top #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `NUM_THREADS
) (
    input wire                              clk,
    input wire                              reset,
    
    input wire                              dcr_write_valid,
    input wire [`VX_DCR_ADDR_WIDTH-1:0]     dcr_write_addr,
    input wire [`VX_DCR_DATA_WIDTH-1:0]     dcr_write_data,

    input  wire                             rop_req_valid,    
    input  wire [`UP(`UUID_BITS)-1:0]       rop_req_uuid,
    input  wire [NUM_LANES-1:0]             rop_req_mask, 
    input  wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] rop_req_pos_x,
    input  wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] rop_req_pos_y,
    input  rgba_t [NUM_LANES-1:0]           rop_req_color,
    input  wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] rop_req_depth,
    input  wire [NUM_LANES-1:0]             rop_req_face,
    output wire                             rop_req_ready,

    output wire [OCACHE_NUM_REQS-1:0]       cache_req_valid,
    output wire [OCACHE_NUM_REQS-1:0]       cache_req_rw,
    output wire [OCACHE_NUM_REQS-1:0][OCACHE_WORD_SIZE-1:0] cache_req_byteen,
    output wire [OCACHE_NUM_REQS-1:0][OCACHE_ADDR_WIDTH-1:0] cache_req_addr,
    output wire [OCACHE_NUM_REQS-1:0][OCACHE_WORD_SIZE*8-1:0] cache_req_data,
    output wire [OCACHE_NUM_REQS-1:0][OCACHE_TAG_WIDTH-1:0] cache_req_tag,
    input  wire [OCACHE_NUM_REQS-1:0]       cache_req_ready,

    input wire  [OCACHE_NUM_REQS-1:0]       cache_rsp_valid,
    input wire  [OCACHE_NUM_REQS-1:0][OCACHE_WORD_SIZE*8-1:0] cache_rsp_data,
    input wire  [OCACHE_NUM_REQS-1:0][OCACHE_TAG_WIDTH-1:0] cache_rsp_tag,
    output wire [OCACHE_NUM_REQS-1:0]       cache_rsp_ready
);

    VX_rop_perf_if perf_rop_if();
    
    VX_dcr_write_if dcr_write_if();

    assign dcr_write_if.valid = dcr_write_valid;
    assign dcr_write_if.addr = dcr_write_addr;
    assign dcr_write_if.data = dcr_write_data;

    VX_rop_req_if #(
        .NUM_LANES (NUM_LANES)
    ) rop_req_if();
    
    assign rop_req_if.valid = rop_req_valid;    
    assign rop_req_if.uuid = rop_req_uuid;
    assign rop_req_if.mask = rop_req_mask; 
    assign rop_req_if.pos_x = rop_req_pos_x;
    assign rop_req_if.pos_y = rop_req_pos_y;
    assign rop_req_if.color = rop_req_color;
    assign rop_req_if.depth = rop_req_depth;
    assign rop_req_if.face = rop_req_face;
    assign rop_req_ready = rop_req_if.ready;

    VX_cache_req_if #(
        .NUM_REQS  (OCACHE_NUM_REQS), 
        .WORD_SIZE (OCACHE_WORD_SIZE), 
        .TAG_WIDTH (OCACHE_TAG_WIDTH)
    ) cache_req_if();

    VX_cache_rsp_if #(
        .NUM_REQS  (OCACHE_NUM_REQS), 
        .WORD_SIZE (OCACHE_WORD_SIZE), 
        .TAG_WIDTH (OCACHE_TAG_WIDTH)
    ) cache_rsp_if();

    assign cache_req_valid = cache_req_if.valid;
    assign cache_req_rw = cache_req_if.rw;
    assign cache_req_byteen = cache_req_if.byteen;
    assign cache_req_addr = cache_req_if.addr;
    assign cache_req_data = cache_req_if.data;
    assign cache_req_tag = cache_req_if.tag;
    assign cache_req_if.ready = cache_req_ready;

    assign cache_rsp_if.valid = cache_rsp_valid;
    assign cache_rsp_if.tag = cache_rsp_tag;
    assign cache_rsp_if.data = cache_rsp_data;
    assign cache_rsp_ready = cache_rsp_if.ready;

    VX_rop_unit #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_LANES   (NUM_LANES)
    ) rop_unit (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .perf_rop_if   (perf_rop_if),
    `endif 
        .dcr_write_if  (dcr_write_if),
        .rop_req_if    (rop_req_if),
        .cache_req_if  (cache_req_if),
        .cache_rsp_if  (cache_rsp_if)
    );

endmodule
