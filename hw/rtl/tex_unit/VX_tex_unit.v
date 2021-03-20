`include "VX_platform.vh"
`include "VX_define.vh"

module VX_tex_unit #(  
    parameter CORE_ID = 0
) (
    input wire  clk,
    input wire  reset,    

    // Texture unit <-> Memory Unit
    VX_dcache_core_req_if dcache_req_if,
    VX_dcache_core_rsp_if dcache_rsp_if,

    // Inputs
    VX_tex_req_if   tex_req_if,
    VX_tex_csr_if   tex_csr_if,

    // Outputs
    VX_tex_rsp_if   tex_rsp_if
);

    localparam MEM_REQ_TAGW = `NW_BITS + 32 + 1 + `NR_BITS + `NTEX_BITS;

    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (reset)

    wire                          rsp_valid;
    wire [`NW_BITS-1:0]           rsp_wid;
    wire [`NUM_THREADS-1:0]       rsp_tmask;
    wire [31:0]                   rsp_PC;
    wire [`NR_BITS-1:0]           rsp_rd;   
    wire                          rsp_wb; 
    wire [`NUM_THREADS-1:0][31:0] rsp_data;    
    wire stall_in, stall_out;

    reg [`TEX_ADDR_BITS-1:0]   tex_addr   [`NUM_TEX_UNITS-1: 0]; 
    reg [`TEX_FMT_BITS-1:0]    tex_format [`NUM_TEX_UNITS-1: 0];
    reg [`TEX_WIDTH_BITS-1:0]  tex_width  [`NUM_TEX_UNITS-1: 0];
    reg [`TEX_HEIGHT_BITS-1:0] tex_height [`NUM_TEX_UNITS-1: 0];
    reg [`TEX_STRIDE_BITS-1:0] tex_stride [`NUM_TEX_UNITS-1: 0];
    reg [`TEX_WRAP_BITS-1:0]   tex_wrap_u [`NUM_TEX_UNITS-1: 0];
    reg [`TEX_WRAP_BITS-1:0]   tex_wrap_v [`NUM_TEX_UNITS-1: 0];
    reg [`TEX_FILTER_BITS-1:0] tex_filter [`NUM_TEX_UNITS-1: 0];

    // CSRs programming

    for (genvar i = 0; i < `NUM_TEX_UNITS; ++i) begin
        always @(posedge clk ) begin        
            if (reset) begin
                tex_addr[i]   <= 0;
                tex_format[i] <= 0;
                tex_width[i]  <= 0;
                tex_height[i] <= 0;
                tex_stride[i] <= 0;
                tex_wrap_u[i] <= 0;
                tex_wrap_v[i] <= 0;
                tex_filter[i] <= 0;
            end begin
                if (tex_csr_if.write_enable) begin            
                    case (tex_csr_if.write_addr)
                        `CSR_TEX_ADDR(i)   : tex_addr[i]   <= tex_csr_if.write_data;
                        `CSR_TEX_FORMAT(i) : tex_format[i] <= tex_csr_if.write_data;
                        `CSR_TEX_WIDTH(i)  : tex_width[i]  <= tex_csr_if.write_data;
                        `CSR_TEX_HEIGHT(i) : tex_height[i] <= tex_csr_if.write_data;
                        `CSR_TEX_STRIDE(i) : tex_stride[i] <= tex_csr_if.write_data;
                        `CSR_TEX_WRAP_U(i) : tex_wrap_u[i] <= tex_csr_if.write_data;
                        `CSR_TEX_WRAP_V(i) : tex_wrap_v[i] <= tex_csr_if.write_data;
                        `CSR_TEX_FILTER(i) : tex_filter[i] <= tex_csr_if.write_data;
                        default:
                            assert(tex_csr_if.write_addr >= `CSR_TEX_BEGIN(0) 
                                && tex_csr_if.write_addr < `CSR_TEX_BEGIN(`CSR_TEX_STATES));
                    endcase
                end
            end
        end
    end

    // address generation

    wire [3:0] mem_req_valid;
    wire [3:0][31:0] mem_req_addr;
    wire [TAG_IN_WIDTH-1:0] mem_req_tag;
    wire mem_req_ready;

    wire mem_rsp_valid;
    wire [3:0][31:0] mem_rsp_data;
    wire [TAG_IN_WIDTH-1:0] mem_rsp_tag;
    wire mem_rsp_ready;
                
    VX_tex_addr_gen #(
        .FRAC_BITS(20)
    ) tex_addr_gen (
        .clk            (clk),
        .reset          (reset),

        .valid_in       (tex_req_if.valid),
        .ready_in       (tex_req_if.ready),   

        .req_tag        ({tex_req_if.wid, tex_req_if.PC, tex_req_if.rd, tex_req_if.wb}),
        .filter         (tex_filter[tex_req_if.unit]),
        .wrap_u         (tex_wrap_ufilter[tex_req_if.unit]),
        .wrap_v         (tex_wrap_v[tex_req_if.unit]),

        .base_addr      (tex_addr[tex_req_if.unit]),
        .log2_stride    (tex_stride[tex_req_if.unit]),
        .log2_width     (tex_width[tex_req_if.unit]),
        .log2_height    (tex_height[tex_req_if.unit]),
        
        .coord_u        (tex_req_if.u),
        .coord_v        (tex_req_if.v),
        .lod            (tex_req_if.lod),

        .mem_req_valid  (mem_req_valid),   
        .mem_req_tag    (mem_req_tag),
        .mem_req_addr   (mem_req_addr),
        .mem_req_ready  (mem_req_ready)
    );

    // retrieve texel values from memory
    
    VX_tex_memory #(
        .CORE_ID       (CORE_ID),
        .REQ_TAG_WIDTH (MEM_REQ_TAGW)
    ) tex_memory (
        .clk           (clk),
        .reset         (reset),

        // memory interface
        .dcache_req_if (dcache_req_if),
        .dcache_rsp_if (dcache_rsp_if),

        // inputs
        req_valid (mem_req_valid),
        req_addr  (mem_req_addr),
        req_tag   (mem_req_tag),
        req_ready (mem_req_ready),

        // outputs
        rsp_valid (mem_rsp_valid),
        rsp_texel (mem_rsp_data),
        rsp_tag   (mem_rsp_tag),
        rsp_ready (mem_rsp_ready)
    );

    // apply sampler

     VX_tex_sampler #(
        .CORE_ID (CORE_ID)
     ) tex_sampler (
        .clk        (clk),
        .reset      (reset)

        // inputs
        //.valid_in   (mem_rsp_valid),
        //.texel      (mem_rsp_data),
        //.req_wid    (mem_rsp_tag),
        //.req_PC     (mem_rsp_tag),
        //.format     (mem_rsp_tag),
        //.ready_in   (mem_rsp_ready),           
    );

    assign tex_req_if.ready = (& pt_addr_ready);

    assign lsu_req_if.valid = (& pt_addr_valid);

    assign lsu_req_if.wid   = tex_req_if.wid;
    assign lsu_req_if.tmask = tex_req_if.tmask;
    assign lsu_req_if.PC    = tex_req_if.PC;
    assign lsu_req_if.rd    = tex_req_if.rd;
    assign lsu_req_if.wb    = tex_req_if.wb;
    assign lsu_req_if.offset = 32'h0000;
    assign lsu_req_if.op_type = `OP_BITS'({1'b0, 3'b000}); //func3 for word load??
    assign lsu_req_if.store_data = {`NUM_THREADS{32'h0000}};

    // wait buffer for fragments  / replace with cache/state fragment fifo for bilerp
    // no filtering for point sampling -> directly from dcache to output response

    assign rsp_valid = ld_commit_if.valid;
    assign rsp_wid   = ld_commit_if.wid;
    assign rsp_tmask = ld_commit_if.tmask;
    assign rsp_PC    = ld_commit_if.PC;
    assign rsp_rd    = ld_commit_if.rd;
    assign rsp_wb    = ld_commit_if.wb;
    assign rsp_data  = ld_commit_if.data; 

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({rsp_valid,        rsp_wid,        rsp_tmask,        rsp_PC,        rsp_rd,        rsp_wb,        rsp_data}),
        .data_out ({tex_rsp_if.valid, tex_rsp_if.wid, tex_rsp_if.tmask, tex_rsp_if.PC, tex_rsp_if.rd, tex_rsp_if.wb, tex_rsp_if.data})
    );

    // output
    assign stall_out = ~tex_rsp_if.ready && tex_rsp_if.valid;

    // can accept new request?
    assign stall_in  = stall_out;

    assign ld_commit_if.ready = ~stall_in;

`ifdef DBG_PRINT_TEX
    always @(posedge clk) begin
        if (tex_csr_if.write_enable 
         && (tex_csr_if.write_addr >= `CSR_TEX_BEGIN(0) 
          && tex_csr_if.write_addr < `CSR_TEX_BEGIN(`CSR_TEX_STATES))) begin
            $display("%t: core%0d-tex_csr: csr_tex0_addr, csr_data=%0h", $time, CORE_ID, tex_addr[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_format, csr_data=%0h", $time, CORE_ID, tex_format[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_width, csr_data=%0h", $time, CORE_ID, tex_width[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_height, csr_data=%0h", $time, CORE_ID, tex_height[0]);
            $display("%t: core%0d-tex_csr: CSR_TEX0_PITCH, csr_data=%0h", $time, CORE_ID, tex_stride[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_wrap_u, csr_data=%0h", $time, CORE_ID, tex_wrap_u[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_wrap_v, csr_data=%0h", $time, CORE_ID, tex_wrap_v[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_min_filter, csr_data=%0h", $time, CORE_ID, tex_min_filter[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_max_filter, csr_data=%0h", $time, CORE_ID, tex_max_filter[0]);
        end
    end
`endif

endmodule