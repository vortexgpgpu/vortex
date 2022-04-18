`include "VX_rop_define.vh"



module VX_rop_slice #(
    parameter CLUSTER_ID = 0,
    parameter NUM_LANES  = 4
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_rop_perf_if.master rop_perf_if,
`endif
   
    // DCRs
    input rop_dcrs_t dcrs,

    // Memory interface
    VX_cache_req_if.master cache_req_if,
    VX_cache_rsp_if.slave  cache_rsp_if,

    // Inputs
    VX_rop_req_if.slave rop_req_if
);
    localparam MEM_TAG_WIDTH = NUM_LANES * (`ROP_DIM_BITS + `ROP_DIM_BITS + 32 + `ROP_DEPTH_BITS + 1);
    localparam DS_TAG_WIDTH = NUM_LANES * (`ROP_DIM_BITS + `ROP_DIM_BITS + 1 + 1 + 32);
    localparam BLEND_TAG_WIDTH  = NUM_LANES * (`ROP_DIM_BITS + `ROP_DIM_BITS + 1);

    wire                                    mem_req_valid, mem_req_valid_r;
    wire [NUM_LANES-1:0]                    mem_req_mask, mem_req_mask_r;
    wire [NUM_LANES-1:0]                    mem_req_ds_pass, mem_req_ds_pass_r;
    wire                                    mem_req_rw, mem_req_rw_r;
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] mem_req_pos_x, mem_req_pos_x_r;
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] mem_req_pos_y, mem_req_pos_y_r;
    rgba_t [NUM_LANES-1:0]                  mem_req_color, mem_req_color_r;
    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] mem_req_depth, mem_req_depth_r;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] mem_req_stencil, mem_req_stencil_r;
    wire [NUM_LANES-1:0]                    mem_req_backface, mem_req_backface_r;
    wire [MEM_TAG_WIDTH-1:0]                mem_req_tag, mem_req_tag_r;
    wire                                    mem_req_ready, mem_req_ready_r;

    wire                                    mem_rsp_valid;
    wire [NUM_LANES-1:0]                    mem_rsp_mask;
    rgba_t [NUM_LANES-1:0]                  mem_rsp_color;
    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] mem_rsp_depth;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] mem_rsp_stencil;
    wire [MEM_TAG_WIDTH-1:0]                mem_rsp_tag;
    wire                                    mem_rsp_ready;

    VX_rop_mem #(
        .CLUSTER_ID (CLUSTER_ID),
        .NUM_LANES  (NUM_LANES),
        .TAG_WIDTH  (MEM_TAG_WIDTH)
    ) rop_mem (
        .clk            (clk),
        .reset          (reset),

    `ifdef PERF_ENABLE
        .rop_perf_if    (rop_perf_if),
    `endif

        .dcrs           (dcrs),

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
        .req_backface   (mem_req_backface_r),
        .req_tag        (mem_req_tag_r),
        .req_ready      (mem_req_ready_r),

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

    wire [NUM_LANES-1:0]    ds_backface;

    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]   ds_depth_ref;
    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]   ds_depth_val;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] ds_stencil_val;

    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0]   ds_depth_out;      
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] ds_stencil_out;
    wire [NUM_LANES-1:0]                        ds_test_out;

    wire [NUM_LANES-1:0][`ROP_DEPTH_FUNC_BITS-1:0] stencil_func;    
    wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_zpass;
    wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_zfail;
    wire [NUM_LANES-1:0][`ROP_STENCIL_OP_BITS-1:0] stencil_fail;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]    stencil_ref;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]    stencil_mask;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0]    stencil_writemask;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign stencil_func[i]  = ds_backface[i] ? dcrs.stencil_back_func  : dcrs.stencil_front_func;    
        assign stencil_zpass[i] = ds_backface[i] ? dcrs.stencil_back_zpass : dcrs.stencil_front_zpass;
        assign stencil_zfail[i] = ds_backface[i] ? dcrs.stencil_back_zfail : dcrs.stencil_front_zfail;
        assign stencil_fail[i]  = ds_backface[i] ? dcrs.stencil_back_fail  : dcrs.stencil_front_fail;
        assign stencil_ref[i]   = ds_backface[i] ? dcrs.stencil_back_ref   : dcrs.stencil_front_ref;
        assign stencil_mask[i]  = ds_backface[i] ? dcrs.stencil_back_mask  : dcrs.stencil_front_mask;
        assign stencil_writemask[i] = ds_backface[i] ? dcrs.stencil_back_writemask  : dcrs.stencil_front_writemask;
    end

    VX_rop_ds #(
        .CLUSTER_ID (CLUSTER_ID),
        .NUM_LANES  (NUM_LANES),
        .TAG_WIDTH  (DS_TAG_WIDTH)
    ) rop_ds (
        .clk            (clk),
        .reset          (reset),

        .valid_in       (ds_valid_in),      
        .tag_in         (ds_tag_in), 
        .ready_in       (ds_ready_in), 

        .valid_out      (ds_valid_out),
        .tag_out        (ds_tag_out),
        .ready_out      (ds_ready_out),
        
        .depth_func     (dcrs.depth_func),
        .depth_writemask(dcrs.depth_writemask),
        .stencil_func   (stencil_func),    
        .stencil_zpass  (stencil_zpass),
        .stencil_zfail  (stencil_zfail),
        .stencil_fail   (stencil_fail),
        .stencil_ref    (stencil_ref),
        .stencil_mask   (stencil_mask),
        .stencil_writemask(stencil_writemask),

        .depth_ref      (ds_depth_ref),
        .depth_val      (ds_depth_val),
        .stencil_val    (ds_stencil_val),    

        .depth_out      (ds_depth_out),        
        .stencil_out    (ds_stencil_out),
        .test_out       (ds_test_out)
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
        .CLUSTER_ID (CLUSTER_ID),
        .NUM_LANES  (NUM_LANES),
        .TAG_WIDTH  (BLEND_TAG_WIDTH)
    ) rop_blend (
        .clk            (clk),
        .reset          (reset),

        .valid_in       (blend_valid_in),      
        .tag_in         (blend_tag_in),
        .ready_in       (blend_ready_in), 

        .valid_out      (blend_valid_out),
        .tag_out        (blend_tag_out),
        .ready_out      (blend_ready_out),

        .blend_mode_rgb (dcrs.blend_mode_rgb),
        .blend_mode_a   (dcrs.blend_mode_a),
        .blend_src_rgb  (dcrs.blend_src_rgb),
        .blend_src_a    (dcrs.blend_src_a),
        .blend_dst_rgb  (dcrs.blend_dst_rgb),
        .blend_dst_a    (dcrs.blend_dst_a),
        .blend_const    (dcrs.blend_const),
        .logic_op       (dcrs.logic_op),
        
        .src_color      (blend_src_color),
        .dst_color      (blend_dst_color),
        .color_out      (blend_color_out)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire color_writeen = (dcrs.cbuf_writemask != 0);

    wire depth_enable  = dcrs.depth_enable;
    wire depth_writeen = dcrs.depth_enable && (dcrs.depth_writemask != 0);

    wire stencil_enable  = dcrs.stencil_back_enable | dcrs.stencil_front_enable;
    wire stencil_writeen = (dcrs.stencil_back_enable && (dcrs.stencil_back_writemask != 0))
                         | (dcrs.stencil_front_enable && (dcrs.stencil_front_writemask != 0));

    wire ds_enable  = depth_enable | stencil_enable;
    wire ds_writeen = depth_writeen | stencil_writeen;

    wire blend_enable  = dcrs.blend_enable;
    wire blend_writeen = dcrs.blend_enable & color_writeen;

    wire mem_readen = blend_enable | ds_enable;

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] mem_rsp_pos_x, mem_rsp_pos_y;

    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] ds_write_pos_x, ds_write_pos_y;
    wire [NUM_LANES-1:0] ds_write_mask, ds_write_backface;
    rgba_t [NUM_LANES-1:0] ds_write_color;

    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] blend_write_pos_x, blend_write_pos_y;
    wire [NUM_LANES-1:0] blend_write_mask;
    
    assign mem_req_tag = {rop_req_if.pos_x, rop_req_if.pos_y, rop_req_if.color, rop_req_if.depth, rop_req_if.backface};
    assign {mem_rsp_pos_x, mem_rsp_pos_y, blend_src_color, ds_depth_ref, ds_backface} = mem_readen ? mem_rsp_tag : mem_req_tag;

    assign ds_tag_in = {mem_rsp_pos_x, mem_rsp_pos_y, mem_rsp_mask, ds_backface, blend_src_color};
    assign {ds_write_pos_x, ds_write_pos_y, ds_write_mask, ds_write_backface, ds_write_color} = ds_tag_out;

    assign blend_tag_in = {mem_rsp_pos_x, mem_rsp_pos_y, mem_rsp_mask};
    assign {blend_write_pos_x, blend_write_pos_y, blend_write_mask} = blend_tag_out;

    wire blend_ds_read = mem_readen && rop_req_if.valid;

    wire blend_ds_write = (ds_writeen && blend_writeen) ? (ds_valid_out && blend_valid_out) :
                            (ds_writeen ? ds_valid_out :
                                (blend_writeen ? blend_valid_out :
                                    1'b0));

    wire write_bypass = !ds_enable && !blend_enable && color_writeen && rop_req_if.valid;

    assign mem_req_valid    = blend_ds_write || blend_ds_read || write_bypass;
    assign mem_req_mask     = blend_ds_write ? (ds_enable ? ds_write_mask : blend_write_mask) : rop_req_if.tmask;
    assign mem_req_ds_pass  = ds_enable ? ds_test_out : {NUM_LANES{1'b1}};
    assign mem_req_rw       = blend_ds_write || write_bypass;
    assign mem_req_backface = blend_ds_write ? ds_write_backface : rop_req_if.backface;
    assign mem_req_pos_x    = blend_ds_write ? (ds_enable ? ds_write_pos_x : blend_write_pos_x) : rop_req_if.pos_x;
    assign mem_req_pos_y    = blend_ds_write ? (ds_enable ? ds_write_pos_y : blend_write_pos_y) : rop_req_if.pos_y;
    assign mem_req_color    = blend_enable ? blend_color_out : (ds_enable ? ds_write_color : rop_req_if.color);
    assign mem_req_depth    = ds_depth_out;
    assign mem_req_stencil  = ds_stencil_out;
    
    assign ds_ready_out     = mem_req_ready && (~blend_enable || blend_valid_out);
    assign blend_ready_out  = mem_req_ready && (~ds_enable || ds_valid_out);
    assign rop_req_if.ready = mem_req_ready && ((!ds_enable && !blend_enable) || ~blend_ds_write);

    assign ds_valid_in      = ds_enable && mem_rsp_valid && (~blend_enable || blend_ready_in);
    assign blend_valid_in   = blend_enable && mem_rsp_valid & (~ds_enable || ds_ready_in);
    assign blend_dst_color  = mem_rsp_color;    

    assign ds_depth_val     = mem_rsp_depth;
    assign ds_stencil_val   = mem_rsp_stencil;    
    assign mem_rsp_ready    = (ds_enable && blend_enable) ? (ds_ready_in && blend_ready_in) :
                                (ds_enable ? ds_ready_in :
                                    (blend_enable ? blend_ready_in :
                                        1'b0));

    wire mem_req_stall = mem_req_valid_r & ~mem_req_ready_r;

    VX_pipe_register #(
        .DATAW	(1 + 1 + NUM_LANES * (1 + 1 + 2 * `ROP_DIM_BITS + $bits(rgba_t) + `ROP_DEPTH_BITS + `ROP_STENCIL_BITS + 1) + MEM_TAG_WIDTH),
        .RESETW (1)
    ) mem_req_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable	  (mem_req_stall),
        .data_in  ({mem_req_valid,   mem_req_rw,   mem_req_mask,   mem_req_ds_pass,   mem_req_pos_x,   mem_req_pos_y,   mem_req_color,   mem_req_depth,   mem_req_stencil,   mem_req_backface,   mem_req_tag}),
        .data_out ({mem_req_valid_r, mem_req_rw_r, mem_req_mask_r, mem_req_ds_pass_r, mem_req_pos_x_r, mem_req_pos_y_r, mem_req_color_r, mem_req_depth_r, mem_req_stencil_r, mem_req_backface_r, mem_req_tag_r})
    );

    assign mem_req_ready = ~mem_req_stall;

endmodule
