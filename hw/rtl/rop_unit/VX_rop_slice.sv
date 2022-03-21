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
    VX_dcache_req_if.master cache_req_if,
    VX_dcache_rsp_if.slave  cache_rsp_if,

    // Inputs
    VX_rop_req_if.slave rop_req_if
);
    localparam MEM_TAG_WIDTH = NUM_LANES * (`ROP_DIM_BITS + `ROP_DIM_BITS + 32 + `ROP_DEPTH_BITS + 1);
    localparam DS_TAG_WIDTH  = NUM_LANES * (`ROP_DIM_BITS + `ROP_DIM_BITS + 1);

    wire                                    mem_req_valid;
    wire [NUM_LANES-1:0]                    mem_req_tmask;
    wire                                    mem_req_rw;
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] mem_req_pos_x;
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] mem_req_pos_y;
    rgba_t [NUM_LANES-1:0]                  mem_req_color;
    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] mem_req_depth;
    wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] mem_req_stencil;
    wire [MEM_TAG_WIDTH-1:0]                mem_req_tag;
    wire                                    mem_req_ready;

    wire                                    mem_rsp_valid;
    wire [NUM_LANES-1:0]                    mem_rsp_tmask;
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

        .req_valid      (mem_req_valid),
        .req_tmask      (mem_req_tmask),
        .req_rw         (mem_req_rw),
        .req_pos_x      (mem_req_pos_x),
        .req_pos_y      (mem_req_pos_y),
        .req_color      (mem_req_color), 
        .req_depth      (mem_req_depth),
        .req_stencil    (mem_req_stencil),
        .req_tag        (mem_req_tag),
        .req_ready      (mem_req_ready),

        .rsp_valid      (mem_rsp_valid),
        .rsp_tmask      (mem_rsp_tmask),
        .rsp_color      (mem_rsp_color), 
        .rsp_depth      (mem_rsp_depth),
        .rsp_stencil    (mem_rsp_stencil),
        .rsp_tag        (mem_rsp_tag),
        .rsp_ready      (mem_rsp_ready)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0]    ds_backface;

    wire                    ds_valid_in;
    wire [DS_TAG_WIDTH-1:0] ds_tag_in;
    wire                    ds_ready_in;   
    wire                    ds_valid_out;
    wire [DS_TAG_WIDTH-1:0] ds_tag_out;
    wire                    ds_ready_out;

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

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign stencil_func[i]  = ds_backface[i] ? dcrs.stencil_back_func  : dcrs.stencil_front_func;    
        assign stencil_zpass[i] = ds_backface[i] ? dcrs.stencil_back_zpass : dcrs.stencil_front_zpass;
        assign stencil_zfail[i] = ds_backface[i] ? dcrs.stencil_back_zfail : dcrs.stencil_front_zfail;
        assign stencil_fail[i]  = ds_backface[i] ? dcrs.stencil_back_fail  : dcrs.stencil_front_fail;
        assign stencil_ref[i]   = ds_backface[i] ? dcrs.stencil_back_ref   : dcrs.stencil_front_ref;
        assign stencil_mask[i]  = ds_backface[i] ? dcrs.stencil_back_mask  : dcrs.stencil_front_mask;
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
        .stencil_writemask(dcrs.stencil_writemask),

        .depth_ref      (ds_depth_ref),
        .depth_val      (ds_depth_val),
        .stencil_val    (ds_stencil_val),    

        .depth_out      (ds_depth_out),        
        .stencil_out    (ds_stencil_out),
        .test_out       (ds_test_out)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire                    blend_valid_in;
    wire                    blend_ready_in;   
    wire                    blend_valid_out;
    wire                    blend_ready_out;

    rgba_t [NUM_LANES-1:0]  blend_src_color;
    rgba_t [NUM_LANES-1:0]  blend_dst_color;
    rgba_t [NUM_LANES-1:0]  blend_color_out;

    VX_rop_blend #(
        .CLUSTER_ID (CLUSTER_ID),
        .NUM_LANES  (NUM_LANES)
    ) rop_blend (
        .clk            (clk),
        .reset          (reset),

        .valid_in       (blend_valid_in),      
        `UNUSED_PIN     (tag_in),
        .ready_in       (blend_ready_in), 

        .valid_out      (blend_valid_out),
        `UNUSED_PIN     (tag_out),
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

    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] mem_rsp_pos_x, mem_write_pos_x;
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] mem_rsp_pos_y, mem_write_pos_y;
    wire [NUM_LANES-1:0] mem_write_tmask;

    wire write_enable = ds_valid_out & blend_valid_out;

    assign mem_req_tag = {rop_req_if.pos_x, rop_req_if.pos_y, rop_req_if.color, rop_req_if.depth, rop_req_if.backface};
    assign {mem_rsp_pos_x, mem_rsp_pos_y, blend_src_color, ds_depth_ref, ds_backface} = mem_rsp_tag;

    assign {mem_write_pos_x, mem_write_pos_y, mem_write_tmask} = ds_tag_out;
    assign ds_tag_in = {mem_rsp_pos_x, mem_rsp_pos_y, mem_rsp_tmask};

    assign mem_req_valid    = write_enable | rop_req_if.valid;
    assign mem_req_tmask    = write_enable ? (ds_test_out & mem_write_tmask) : rop_req_if.tmask;
    assign mem_req_rw       = write_enable;
    assign mem_req_pos_x    = write_enable ? mem_write_pos_x : rop_req_if.pos_x;
    assign mem_req_pos_y    = write_enable ? mem_write_pos_y : rop_req_if.pos_y;
    assign mem_req_color    = blend_color_out;
    assign mem_req_depth    = ds_depth_out;
    assign mem_req_stencil  = ds_stencil_out;    
    assign ds_ready_out     = mem_req_ready & blend_valid_out;
    assign blend_ready_out  = mem_req_ready & ds_valid_out;
    assign rop_req_if.ready = mem_req_ready & ~write_enable;

    assign ds_valid_in      = mem_rsp_valid & blend_ready_in;
    assign blend_valid_in   = mem_rsp_valid & ds_ready_in;    
    assign blend_dst_color  = mem_rsp_color;    
    assign ds_depth_val     = mem_rsp_depth;
    assign ds_stencil_val   = mem_rsp_stencil;    
    assign mem_rsp_ready    = ds_ready_in & blend_ready_in;

endmodule
