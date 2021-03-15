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
    // VX_commit_if    gpu_commit_if
    // // Texture Request
    // input wire                  tex_req_valid,       
    // input wire [`TADDRW-1:0]     tex_req_u,
    // input wire [`TADDRW-1:0]     tex_req_v,
    // input wire [`MADDRW-1:0]     tex_req_addr,    
    // input wire [`MAXWTW-1:0]     tex_req_width,
    // input wire [`MAXHTW-1:0]     tex_req_height,
    // input wire [`MAXFTW-1:0]     tex_req_format,
    // input wire [`MAXFMW-1:0]     tex_req_filter,
    // input wire [`MAXAMW-1:0]     tex_req_clamp,
    // input wire [`TAGW-1:0]       tex_req_tag,
    // output wire                 tex_req_ready,  

    // // Texture Response
    // output wire                 tex_rsp_valid,   
    // output wire [`TAGW-1:0]      tex_rsp_tag,
    // input wire  [`DATAW-1:0]     tex_rsp_data,
    // input  wire                 tex_rsp_ready,

    // Cache Request
    // output wire [NUMCRQS-1:0]             cache_req_valids, 
    // output wire [NUMCRQS-1:0][MADDRW-1:0] cache_req_addrs,
    // input  wire                           cache_req_ready,

    // Cache Response
    // input  wire              cache_rsp_valid,
    // input  wire [MADDRW-1:0] cache_rsp_addr,
    // input  wire [DATAW-1:0]  cache_rsp_data,
    // output wire              cache_rsp_ready
);

    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (reset)

    `UNUSED_VAR(tex_addr)
    `UNUSED_VAR(tex_format)
    `UNUSED_VAR(tex_width)
    `UNUSED_VAR(tex_height)
    `UNUSED_VAR(tex_stride)
    `UNUSED_VAR(tex_wrap_u)
    `UNUSED_VAR(tex_wrap_v)
    `UNUSED_VAR(tex_min_filter)
    `UNUSED_VAR(tex_max_filter)

    reg [`CSR_WIDTH-1:0] tex_addr [`NUM_TEX_UNITS-1: 0]; 
    reg [`CSR_WIDTH-1:0] tex_format [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_width [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_height [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_stride [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_wrap_u [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_wrap_v [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_min_filter [`NUM_TEX_UNITS-1: 0];
    reg [`CSR_WIDTH-1:0] tex_max_filter [`NUM_TEX_UNITS-1: 0];

    //tex csr programming, need to make make consistent with `NUM_TEX_UNITS
    always @(posedge clk ) begin
        if (tex_csr_if.write_enable) begin
            case (tex_csr_if.write_addr)
                `CSR_TEX0_ADDR : tex_addr[0] <= tex_csr_if.write_data;
                `CSR_TEX0_FORMAT : tex_format[0] <= tex_csr_if.write_data;
                `CSR_TEX0_WIDTH : tex_width[0] <= tex_csr_if.write_data;
                `CSR_TEX0_HEIGHT : tex_height[0] <= tex_csr_if.write_data;
                `CSR_TEX0_STRIDE : tex_stride[0] <= tex_csr_if.write_data;
                `CSR_TEX0_WRAP_U : tex_wrap_u[0] <= tex_csr_if.write_data;
                `CSR_TEX0_WRAP_V : tex_wrap_v[0] <= tex_csr_if.write_data;
                `CSR_TEX0_MIN_FILTER : tex_min_filter[0] <= tex_csr_if.write_data;
                `CSR_TEX0_MAX_FILTER : tex_max_filter[0] <= tex_csr_if.write_data;

                `CSR_TEX1_ADDR : tex_addr[1] <= tex_csr_if.write_data;
                `CSR_TEX1_FORMAT : tex_format[1] <= tex_csr_if.write_data;
                `CSR_TEX1_WIDTH : tex_width[1] <= tex_csr_if.write_data;
                `CSR_TEX1_HEIGHT : tex_height[1] <= tex_csr_if.write_data;
                `CSR_TEX1_STRIDE : tex_stride[1] <= tex_csr_if.write_data;
                `CSR_TEX1_WRAP_U : tex_wrap_u[1] <= tex_csr_if.write_data;
                `CSR_TEX1_WRAP_V : tex_wrap_v[1] <= tex_csr_if.write_data;
                `CSR_TEX1_MIN_FILTER : tex_min_filter[1] <= tex_csr_if.write_data;
                `CSR_TEX1_MAX_FILTER : tex_max_filter[1] <= tex_csr_if.write_data;
                default: 
                    assert(tex_csr_if.write_addr > `CSR_TEX_END || tex_csr_if.write_addr < `CSR_TEX_BEGIN) else $error("%t: invalid CSR write address: %0h", $time, tex_csr_if.write_addr);
            endcase
        end
    end

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign tex_rsp_if.data[i] = 32'hFAAF;
    end   

    assign tex_rsp_if.ready = 1'b1;

    `ifdef DBG_PRINT_TEX_CSRS
    always @(posedge clk) begin
        if (tex_csr_if.write_addr <= `CSR_TEX_END || tex_csr_if.write_addr >= `CSR_TEX_BEGIN) begin
            $display("%t: core%0d-tex_csr: csr_tex0_addr, csr_data=%0h", $time, CORE_ID, tex_addr[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_format, csr_data=%0h", $time, CORE_ID, tex_format[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_width, csr_data=%0h", $time, CORE_ID, tex_width[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_height, csr_data=%0h", $time, CORE_ID, tex_height[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_stride, csr_data=%0h", $time, CORE_ID, tex_stride[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_wrap_u, csr_data=%0h", $time, CORE_ID, tex_wrap_u[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_wrap_v, csr_data=%0h", $time, CORE_ID, tex_wrap_v[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_min_filter, csr_data=%0h", $time, CORE_ID, tex_min_filter[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_max_filter, csr_data=%0h", $time, CORE_ID, tex_max_filter[0]);
        end
    end
    `endif 


endmodule