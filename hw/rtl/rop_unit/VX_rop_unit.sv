`include "VX_rop_define.vh"

module VX_rop_unit #(
    parameter string INSTANCE_ID = "",
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
    localparam MEM_TAG_WIDTH = `UP(`UUID_BITS) + NUM_LANES * (`ROP_DIM_BITS + `ROP_DIM_BITS + 32 + `ROP_DEPTH_BITS + 1);
    localparam DS_TAG_WIDTH = NUM_LANES * (`ROP_DIM_BITS + `ROP_DIM_BITS + 1 + 1 + 32);
    localparam BLEND_TAG_WIDTH  = NUM_LANES * (`ROP_DIM_BITS + `ROP_DIM_BITS + 1);

    // DCRs

    rop_dcrs_t rop_dcrs;

    VX_rop_dcr rop_dcr (
        .clk        (clk),
        .reset      (reset),
        .dcr_write_if(dcr_write_if),
        .rop_dcrs   (rop_dcrs)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire                                    mem_req_valid, mem_req_valid_r;
    wire [NUM_LANES-1:0]                    mem_req_mask, mem_req_mask_r;
    wire [NUM_LANES-1:0]                    mem_req_ds_pass, mem_req_ds_pass_r;
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

    VX_rop_mem #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (MEM_TAG_WIDTH)
    ) rop_mem (
        .clk            (clk),
        .reset          (reset),

        .dcrs           (rop_dcrs),

        .cache_req_if   (cache_req_if),
        .cache_rsp_if   (cache_rsp_if),

        .req_valid      (mem_req_valid_r),
        .req_mask       (mem_req_mask_r),
        .req_ds_pass    (mem_req_ds_pass_r),
        .req_rw         (mem_req_rw_r),
        .req_pos_x      (mem_req_pos_x_r),
        .req_pos_y      (mem_req_pos_y_r),
        .req_color      (mem_req_color_r), 
        .req_depth      (mem_req_depth_r),
        .req_stencil    (mem_req_stencil_r),
        .req_face       (mem_req_face_r),
        .req_tag        (mem_req_tag_r),
        .req_ready      (mem_req_ready_r),

        .rsp_valid      (mem_rsp_valid),
        .rsp_mask       (mem_rsp_mask),
        .rsp_color      (mem_rsp_color), 
        .rsp_depth      (mem_rsp_depth),
        .rsp_stencil    (mem_rsp_stencil),
        .rsp_tag        (mem_rsp_tag),
        .rsp_ready      (mem_rsp_ready),
        .write_notify   (mem_write_notify)
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

    VX_rop_ds #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (DS_TAG_WIDTH)
    ) rop_ds (
        .clk            (clk),
        .reset          (reset),

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

    VX_rop_blend #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (BLEND_TAG_WIDTH)
    ) rop_blend (
        .clk            (clk),
        .reset          (reset),

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
                         | (rop_dcrs.stencil_enable[1] && (rop_dcrs.stencil_writemask[1] != 0));

    wire ds_enable  = depth_enable | stencil_enable;
    wire ds_writeen = depth_writeen | stencil_writeen;

    wire blend_enable  = rop_dcrs.blend_enable;
    wire blend_writeen = rop_dcrs.blend_enable & color_writeen;

    wire mem_readen = ds_enable | blend_enable;

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] mem_rsp_pos_x, mem_rsp_pos_y;
    wire [`UP(`UUID_BITS)-1:0] mem_rsp_uuid;
    `UNUSED_VAR (mem_rsp_uuid)

    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] ds_write_pos_x, ds_write_pos_y;
    wire [NUM_LANES-1:0] ds_write_mask, ds_write_face;
    rgba_t [NUM_LANES-1:0] ds_write_color;

    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] blend_write_pos_x, blend_write_pos_y;
    wire [NUM_LANES-1:0] blend_write_mask;

    wire pending_reads_full;
    
    assign mem_req_tag = {rop_req_if.uuid, rop_req_if.pos_x, rop_req_if.pos_y, rop_req_if.color, rop_req_if.depth, rop_req_if.face};
    assign {mem_rsp_uuid, mem_rsp_pos_x, mem_rsp_pos_y, blend_src_color, ds_depth_ref, ds_face} = mem_rsp_tag;

    assign ds_tag_in = {mem_rsp_pos_x, mem_rsp_pos_y, mem_rsp_mask, ds_face, blend_src_color};
    assign {ds_write_pos_x, ds_write_pos_y, ds_write_mask, ds_write_face, ds_write_color} = ds_tag_out;

    assign blend_tag_in = {mem_rsp_pos_x, mem_rsp_pos_y, mem_rsp_mask};
    assign {blend_write_pos_x, blend_write_pos_y, blend_write_mask} = blend_tag_out;

    wire ds_blend_read = mem_readen && rop_req_if.valid && ~pending_reads_full;

    wire ds_blend_write = (ds_writeen && blend_writeen) ? (ds_valid_out && blend_valid_out) :
                            (ds_writeen ? ds_valid_out :
                                (blend_writeen ? blend_valid_out :
                                    1'b0));

    wire write_bypass = !ds_enable && !blend_enable && color_writeen && rop_req_if.valid;

    assign mem_req_valid    = ds_blend_write || ds_blend_read || write_bypass;
    assign mem_req_mask     = ds_blend_write ? (ds_enable ? ds_write_mask : blend_write_mask) : rop_req_if.mask;
    assign mem_req_ds_pass  = ds_enable ? ds_pass_out : {NUM_LANES{1'b1}};
    assign mem_req_rw       = ds_blend_write || write_bypass;
    assign mem_req_face     = ds_blend_write ? ds_write_face : rop_req_if.face;
    assign mem_req_pos_x    = ds_blend_write ? (ds_enable ? ds_write_pos_x : blend_write_pos_x) : rop_req_if.pos_x;
    assign mem_req_pos_y    = ds_blend_write ? (ds_enable ? ds_write_pos_y : blend_write_pos_y) : rop_req_if.pos_y;
    assign mem_req_color    = blend_enable ? blend_color_out : (ds_enable ? ds_write_color : rop_req_if.color);
    assign mem_req_depth    = ds_depth_out;
    assign mem_req_stencil  = ds_stencil_out;
    
    assign ds_ready_out     = mem_req_ready && (~blend_enable || blend_valid_out);
    assign blend_ready_out  = mem_req_ready && (~ds_enable || ds_valid_out);
    assign rop_req_if.ready = mem_req_ready && ((~ds_enable && ~blend_enable) || ~ds_blend_write) && ~pending_reads_full;

    assign ds_valid_in      = ds_enable && mem_rsp_valid && (~blend_enable || blend_ready_in);
    assign blend_valid_in   = blend_enable && mem_rsp_valid & (~ds_enable || ds_ready_in);
    assign blend_dst_color  = mem_rsp_color;    

    assign ds_depth_val     = mem_rsp_depth;
    assign ds_stencil_val   = mem_rsp_stencil;    
    assign mem_rsp_ready    = (ds_enable && blend_enable) ? (ds_ready_in && blend_ready_in) :
                                (ds_enable ? ds_ready_in :
                                    (blend_enable ? blend_ready_in :
                                        1'b0));

    wire mem_req_fire = mem_req_valid & mem_req_ready;

    // to prevent potential deadlocks, 
    // ensure the memory scheduler's queue doesn't fill up
    VX_pending_size #( 
        .SIZE (`ROP_MEM_QUEUE_SIZE)
    ) pending_reads (
        .clk   (clk),
        .reset (reset),
        .incr  (mem_req_fire && ~mem_req_rw && (ds_writeen || blend_writeen)),
        .decr  (mem_write_notify && (ds_writeen || blend_writeen)),
        .full  (pending_reads_full),
        `UNUSED_PIN (size),
        `UNUSED_PIN (empty)
    );

    VX_generic_buffer #(
        .DATAW	 (1 + NUM_LANES * (1 + 1 + 2 * `ROP_DIM_BITS + $bits(rgba_t) + `ROP_DEPTH_BITS + `ROP_STENCIL_BITS + 1) + MEM_TAG_WIDTH),
        .OUT_REG (1)
    ) mem_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_req_valid),
        .ready_in  (mem_req_ready),
        .data_in   ({mem_req_rw,   mem_req_mask,   mem_req_ds_pass,   mem_req_pos_x,   mem_req_pos_y,   mem_req_color,   mem_req_depth,   mem_req_stencil,   mem_req_face,   mem_req_tag}),
        .data_out  ({mem_req_rw_r, mem_req_mask_r, mem_req_ds_pass_r, mem_req_pos_x_r, mem_req_pos_y_r, mem_req_color_r, mem_req_depth_r, mem_req_stencil_r, mem_req_face_r, mem_req_tag_r}),
        .valid_out (mem_req_valid_r),
        .ready_out (mem_req_ready_r)
    );

`ifdef PERF_ENABLE

    wire [$clog2(OCACHE_NUM_REQS+1)-1:0] perf_mem_rd_req_per_cycle;
    wire [$clog2(OCACHE_NUM_REQS+1)-1:0] perf_mem_wr_req_per_cycle;
    wire [$clog2(OCACHE_NUM_REQS+1)-1:0] perf_mem_rd_rsp_per_cycle;
    wire [$clog2(OCACHE_NUM_REQS+1)+1-1:0] perf_pending_reads_cycle;

    wire [OCACHE_NUM_REQS-1:0] perf_mem_rd_req_per_mask = cache_req_if.valid & ~cache_req_if.rw & cache_req_if.ready;
    wire [OCACHE_NUM_REQS-1:0] perf_mem_wr_req_per_mask = cache_req_if.valid & cache_req_if.rw & cache_req_if.ready;
    wire [OCACHE_NUM_REQS-1:0] perf_mem_rd_rsp_per_mask = cache_rsp_if.valid & cache_rsp_if.ready;

    `POP_COUNT(perf_mem_rd_req_per_cycle, perf_mem_rd_req_per_mask);    
    `POP_COUNT(perf_mem_wr_req_per_cycle, perf_mem_wr_req_per_mask);    
    `POP_COUNT(perf_mem_rd_rsp_per_cycle, perf_mem_rd_rsp_per_mask);

    reg [`PERF_CTR_BITS-1:0] perf_pending_reads;   
    assign perf_pending_reads_cycle = perf_mem_rd_req_per_cycle - perf_mem_rd_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_pending_reads <= 0;
        end else begin
            perf_pending_reads <= perf_pending_reads + `PERF_CTR_BITS'($signed(perf_pending_reads_cycle));
        end
    end

    wire perf_stall_cycle = rop_req_if.valid & ~rop_req_if.ready;

    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_writes;
    reg [`PERF_CTR_BITS-1:0] perf_mem_latency;
    reg [`PERF_CTR_BITS-1:0] perf_stall_cycles;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_reads    <= 0;
            perf_mem_writes   <= 0;
            perf_mem_latency  <= 0;
            perf_stall_cycles <= 0;
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
