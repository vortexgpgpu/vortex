`include "VX_define.vh"

module VX_tex_unit #(
    parameter TADDRW = 32,
    parameter MADDRW = 32,
    parameter DATAW  = 32,
    parameter MAXWTW = 8,
    parameter MAXHTW = 8,
    parameter MAXFTW = 2,
    parameter MAXFMW = 1,
    parameter MAXAMW = 2,
    parameter TAGW   = 16,

    parameter NUMCRQS = 32,
) (
    input wire clk,
    input wire reset,

    // Texture Request
    input wire                  tex_req_valid,       
    input wire [TADDRW-1:0]     tex_req_u,
    input wire [TADDRW-1:0]     tex_req_v,
    input wire [MADDRW-1:0]     tex_req_addr,    
    input wire [MAXWTW-1:0]     tex_req_width,
    input wire [MAXHTW-1:0]     tex_req_height,
    input wire [MAXFTW-1:0]     tex_req_format,
    input wire [MAXFMW-1:0]     tex_req_filter,
    input wire [MAXAMW-1:0]     tex_req_clamp,
    input wire [TAGW-1:0]       tex_req_tag,
    output wire                 tex_req_ready,  

    // Texture Response
    output wire                 tex_rsp_valid,   
    output wire [TAGW-1:0]      tex_rsp_tag,
    input wire  [DATAW-1:0]     tex_rsp_data,
    input  wire                 tex_rsp_ready,

    // Cache Request
    output wire [NUMCRQS-1:0]             cache_req_valids, 
    output wire [NUMCRQS-1:0][MADDRW-1:0] cache_req_addrs,
    input  wire                           cache_req_ready,

    // Cache Response
    input  wire              cache_rsp_valid,
    input  wire [MADDRW-1:0] cache_rsp_addr,
    input  wire [DATAW-1:0]  cache_rsp_data,
    output wire              cache_rsp_ready
);

endmodule