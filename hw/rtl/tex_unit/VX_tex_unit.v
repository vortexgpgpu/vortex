`include "VX_platform.vh"
`include "VX_define.vh"

module VX_tex_unit #(  
    parameter CORE_ID = 0
) (
    input wire  clk,
    input wire  reset,

    // Inputs
    VX_tex_req_if   tex_req_if,
    VX_tex_csr_if   tex_csr_if,

    // Outputs
    VX_tex_rsp_if   tex_rsp_if,

    // Texture unit <-> Memory Unit
    VX_dcache_core_req_if dcache_req_if,
    VX_dcache_core_rsp_if dcache_rsp_if
);

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

    reg [`CSR_WIDTH-1:0] tex_addr [`NUM_TEX_UNITS-1: 0]; 
    reg [`CSR_WIDTH-1:0] tex_format [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_width [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_height [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_stride [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_wrap_u [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_wrap_v [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_min_filter [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_max_filter [`NUM_TEX_UNITS-1: 0];

    `UNUSED_VAR (tex_format)
    `UNUSED_VAR (tex_stride)
    `UNUSED_VAR (tex_wrap_u)
    `UNUSED_VAR (tex_wrap_v)
    `UNUSED_VAR (tex_min_filter)
    `UNUSED_VAR (tex_max_filter)

    //tex csr programming, need to make make consistent with `NUM_TEX_UNITS
    always @(posedge clk ) begin
        if (tex_csr_if.write_enable) begin
            case (tex_csr_if.write_addr)
                `CSR_TEX0_ADDR       : tex_addr[0] <= tex_csr_if.write_data;
                `CSR_TEX0_FORMAT     : tex_format[0] <= tex_csr_if.write_data;
                `CSR_TEX0_WIDTH      : tex_width[0] <= tex_csr_if.write_data;
                `CSR_TEX0_HEIGHT     : tex_height[0] <= tex_csr_if.write_data;
                `CSR_TEX0_PITCH     : tex_stride[0] <= tex_csr_if.write_data;
                `CSR_TEX0_WRAP_U     : tex_wrap_u[0] <= tex_csr_if.write_data;
                `CSR_TEX0_WRAP_V     : tex_wrap_v[0] <= tex_csr_if.write_data;
                `CSR_TEX0_MIN_FILTER : tex_min_filter[0] <= tex_csr_if.write_data;
                `CSR_TEX0_MAX_FILTER : tex_max_filter[0] <= tex_csr_if.write_data;

                `CSR_TEX1_ADDR       : tex_addr[1] <= tex_csr_if.write_data;
                `CSR_TEX1_FORMAT     : tex_format[1] <= tex_csr_if.write_data;
                `CSR_TEX1_WIDTH      : tex_width[1] <= tex_csr_if.write_data;
                `CSR_TEX1_HEIGHT     : tex_height[1] <= tex_csr_if.write_data;
                `CSR_TEX1_PITCH     : tex_stride[1] <= tex_csr_if.write_data;
                `CSR_TEX1_WRAP_U     : tex_wrap_u[1] <= tex_csr_if.write_data;
                `CSR_TEX1_WRAP_V     : tex_wrap_v[1] <= tex_csr_if.write_data;
                `CSR_TEX1_MIN_FILTER : tex_min_filter[1] <= tex_csr_if.write_data;
                `CSR_TEX1_MAX_FILTER : tex_max_filter[1] <= tex_csr_if.write_data;
                default:;
            endcase
        end
    end

    // texture response
    `UNUSED_VAR (tex_req_if.lod)

    // texture unit <-> dcache 
    VX_lsu_req_if   lsu_req_if();
    VX_commit_if    ld_commit_if();

    VX_tex_memory #(
        .CORE_ID(CORE_ID)
    ) tex_memory (
        .clk            (clk),
        .reset          (reset),
        .dcache_req_if  (dcache_req_if),
        .dcache_rsp_if  (dcache_rsp_if),
        .lsu_req_if     (lsu_req_if),
        .ld_commit_if   (ld_commit_if)
    );

    //point sampling - texel address computation
    wire [`NUM_THREADS-1:0] pt_addr_valid;
    wire [`NUM_THREADS-1:0] pt_addr_ready;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire [`CSR_WIDTH-1:0] tex_addr_select;
        wire [`CSR_WIDTH-1:0] tex_width_select;
        wire [`CSR_WIDTH-1:0] tex_height_select;
        
        assign tex_addr_select = (tex_req_if.t[i] == 'b1) ? tex_addr[1] : tex_addr[0];
        assign tex_width_select = (tex_req_if.t[i] == 'b1) ? tex_width[1] : tex_width[0];
        assign tex_height_select = (tex_req_if.t[i] == 'b1) ? tex_height[1] : tex_height[0];
        
        VX_tex_pt_addr #(
            .FRAC_BITS(28)
        ) tex_pt_addr (
            .clk                (clk),
            .reset              (reset),

            .valid_in           (tex_req_if.valid),
            .ready_out          (pt_addr_ready[i]),   

            .tex_addr           (tex_addr_select),
            .tex_width          (tex_width_select),
            .tex_height         (tex_height_select),

            .tex_u              (tex_req_if.u[i]),
            .tex_v              (tex_req_if.v[i]),

            .pt_addr            (lsu_req_if.base_addr[i]),   

            .valid_out          (pt_addr_valid[i]),
            .ready_in           (lsu_req_if.ready)
        );
    end

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
         && (tex_csr_if.write_addr <= `CSR_TEX_END 
          || tex_csr_if.write_addr >= `CSR_TEX_BEGIN)) begin
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