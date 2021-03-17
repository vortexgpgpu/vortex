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
    VX_tex_rsp_if   tex_rsp_if
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

    `UNUSED_VAR (tex_addr)
    `UNUSED_VAR (tex_format)
    `UNUSED_VAR (tex_width)
    `UNUSED_VAR (tex_height)
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
    `UNUSED_VAR (tex_req_if.u)
    `UNUSED_VAR (tex_req_if.v)
    `UNUSED_VAR (tex_req_if.lod_t)

    assign stall_in  = stall_out;

    assign rsp_valid = tex_req_if.valid;
    assign rsp_wid   = tex_req_if.wid;
    assign rsp_tmask = tex_req_if.tmask;
    assign rsp_PC    = tex_req_if.PC;
    assign rsp_rd    = tex_req_if.rd;
    assign rsp_wb    = tex_req_if.wb;
    assign rsp_data  = {`NUM_THREADS{32'hFF0000FF}}; // dummy blue value

    // output
    assign stall_out = ~tex_rsp_if.ready && tex_rsp_if.valid;

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

    // can accept new request?
    assign tex_req_if.ready = ~stall_in;

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