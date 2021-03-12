`include "VX_platform.vh"
`include "VX_define.vh"

module VX_tex_unit #(  
    parameter CORE_ID = 0
) (
    input wire  clk,
    input wire  reset,
    // Inputs
    VX_tex_req_if   tex_req_if,

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

    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign tex_rsp_if.data[i] = 32'hFAAF;
    end   

    assign tex_rsp_if.ready = 1'b1;

endmodule