`ifndef VX_ROP_TYPES_VH
`define VX_ROP_TYPES_VH

`include "VX_define.vh"

package VX_rop_types;

typedef struct packed {
    logic [7:0] a;
    logic [7:0] r;
    logic [7:0] g;
    logic [7:0] b;
} rgba_t;

typedef struct packed {
    logic [31:0]                        cbuf_addr;
    logic [31:0]                        cbuf_pitch;
    logic [3:0]                         cbuf_writemask;

    logic [31:0]                        zbuf_addr;
    logic [31:0]                        zbuf_pitch;

    logic                               depth_enable;
    logic [`ROP_DEPTH_FUNC_BITS-1:0]    depth_func;
    logic                               depth_writemask;
    
    logic                               stencil_front_enable;
    logic [`ROP_DEPTH_FUNC_BITS-1:0]    stencil_front_func;    
    logic [`ROP_STENCIL_OP_BITS-1:0]    stencil_front_zpass;
    logic [`ROP_STENCIL_OP_BITS-1:0]    stencil_front_zfail;
    logic [`ROP_STENCIL_OP_BITS-1:0]    stencil_front_fail;
    logic [`ROP_STENCIL_BITS-1:0]       stencil_front_ref;
    logic [`ROP_STENCIL_BITS-1:0]       stencil_front_mask;    
    logic [`ROP_STENCIL_BITS-1:0]       stencil_front_writemask;

    logic                               stencil_back_enable;
    logic [`ROP_DEPTH_FUNC_BITS-1:0]    stencil_back_func;    
    logic [`ROP_STENCIL_OP_BITS-1:0]    stencil_back_zpass;
    logic [`ROP_STENCIL_OP_BITS-1:0]    stencil_back_zfail;
    logic [`ROP_STENCIL_OP_BITS-1:0]    stencil_back_fail;
    logic [`ROP_STENCIL_BITS-1:0]       stencil_back_ref;
    logic [`ROP_STENCIL_BITS-1:0]       stencil_back_mask;
    logic [`ROP_STENCIL_BITS-1:0]       stencil_back_writemask;
    
    logic                               blend_enable;
    logic [`ROP_BLEND_MODE_BITS-1:0]    blend_mode_rgb;
    logic [`ROP_BLEND_MODE_BITS-1:0]    blend_mode_a;
    logic [`ROP_BLEND_FUNC_BITS-1:0]    blend_src_rgb;
    logic [`ROP_BLEND_FUNC_BITS-1:0]    blend_src_a;
    logic [`ROP_BLEND_FUNC_BITS-1:0]    blend_dst_rgb;
    logic [`ROP_BLEND_FUNC_BITS-1:0]    blend_dst_a;
    rgba_t                              blend_const;
    
    logic [`ROP_LOGIC_OP_BITS-1:0]      logic_op;
} rop_dcrs_t;

typedef struct packed {
    logic [1:0]                 rt_idx;
    logic [`ROP_DIM_BITS-1:0]   pos_x;
    logic [`ROP_DIM_BITS-1:0]   pos_y;    
    logic [23:0]                depth;
    logic [2:0]                 sample_idx;
    logic [7:0]                 sample_mask;
} rop_csrs_t;

typedef struct packed {
    logic                                         valid;
    logic [`NUM_THREADS-1:0]                      tmask; 
    logic [`NUM_THREADS-1:0][`ROP_DIM_BITS-1:0]   pos_x;
    logic [`NUM_THREADS-1:0][`ROP_DIM_BITS-1:0]   pos_y;
    logic [`NUM_THREADS-1:0][31:0]                color;
    logic [`NUM_THREADS-1:0][`ROP_DEPTH_BITS-1:0] depth;
    logic [`NUM_THREADS-1:0]                      backface;
} rop_queue_entry;

task trace_rop_state (
    input int                  level,
    input [`DCR_ADDR_BITS-1:0] state
);
    case (state)
        `DCR_ROP_CBUF_ADDR:         dpi_trace(level, "CBUF_ADDR");
        `DCR_ROP_CBUF_PITCH:        dpi_trace(level, "CBUF_PITCH");
        `DCR_ROP_CBUF_WRITEMASK:    dpi_trace(level, "CBUF_WRITEMASK");
        `DCR_ROP_ZBUF_ADDR:         dpi_trace(level, "ZBUF_ADDR");
        `DCR_ROP_ZBUF_PITCH:        dpi_trace(level, "ZBUF_PITCH");
        `DCR_ROP_DEPTH_FUNC:        dpi_trace(level, "DEPTH_FUNC");
        `DCR_ROP_DEPTH_WRITEMASK:   dpi_trace(level, "DEPTH_WRITEMASK");
        `DCR_ROP_STENCIL_FUNC:      dpi_trace(level, "STENCIL_FUNC");        
        `DCR_ROP_STENCIL_ZPASS:     dpi_trace(level, "STENCIL_ZPASS");
        `DCR_ROP_STENCIL_ZFAIL:     dpi_trace(level, "STENCIL_ZFAIL");
        `DCR_ROP_STENCIL_FAIL:      dpi_trace(level, "STENCIL_FAIL");
        `DCR_ROP_STENCIL_REF:       dpi_trace(level, "STENCIL_REF");
        `DCR_ROP_STENCIL_MASK:      dpi_trace(level, "STENCIL_MASK");        
        `DCR_ROP_STENCIL_WRITEMASK: dpi_trace(level, "STENCIL_WRITEMASK");
        `DCR_ROP_BLEND_MODE:        dpi_trace(level, "BLEND_MODE");
        `DCR_ROP_BLEND_FUNC:        dpi_trace(level, "BLEND_FUNC");
        `DCR_ROP_BLEND_CONST:       dpi_trace(level, "BLEND_CONST");
        `DCR_ROP_LOGIC_OP:          dpi_trace(level, "LOGIC_OP");        
        default:                    dpi_trace(level, "?");
    endcase  
endtask

task trace_rop_csr (
    input int                  level,
    input [`CSR_ADDR_BITS-1:0] addr
);
    case (addr)
        `CSR_ROP_RT_IDX:        dpi_trace(level, "RT_IDX");
        `CSR_ROP_SAMPLE_IDX:    dpi_trace(level, "SAMPLE_IDX");
        default:                dpi_trace(level, "?");
    endcase  
endtask

endpackage

`endif
