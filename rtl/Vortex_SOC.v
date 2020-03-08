`include "VX_define.v"
`include "VX_cache_config.v"

module Vortex_SOC (
    input  wire           clk,
    input  wire           reset,
    input  wire[31:0] icache_response_instruction,
    output wire[31:0] icache_request_pc_address,
    // IO
    output wire        io_valid,
    output wire[31:0]  io_data,

    // DRAM Dcache Req
    output wire                              dram_req,
    output wire                              dram_req_write,
    output wire                              dram_req_read,
    output wire [31:0]                       dram_req_addr,
    output wire [31:0]                       dram_req_size,
    output wire [31:0]                       dram_req_data[`DBANK_LINE_SIZE_RNG],
    output wire [31:0]                       dram_expected_lat,

    // DRAM Dcache Res
    output wire                              dram_fill_accept,
    input  wire                              dram_fill_rsp,
    input  wire [31:0]                       dram_fill_rsp_addr,
    input  wire [31:0]                       dram_fill_rsp_data[`DBANK_LINE_SIZE_RNG],


    // DRAM Icache Req
    output wire                              I_dram_req,
    output wire                              I_dram_req_write,
    output wire                              I_dram_req_read,
    output wire [31:0]                       I_dram_req_addr,
    output wire [31:0]                       I_dram_req_size,
    output wire [31:0]                       I_dram_req_data[`DBANK_LINE_SIZE_RNG],
    output wire [31:0]                       I_dram_expected_lat,

    // DRAM Icache Res
    output wire                              I_dram_fill_accept,
    input  wire                              I_dram_fill_rsp,
    input  wire [31:0]                       I_dram_fill_rsp_addr,
    input  wire [31:0]                       I_dram_fill_rsp_data[`DBANK_LINE_SIZE_RNG],


    output wire        out_ebreak
	);


    Vortex vortex_core(
        .clk                        (clk),
        .reset                      (reset),
        .icache_response_instruction(icache_response_instruction),
        .icache_request_pc_address  (icache_request_pc_address),
        .io_valid                   (io_valid),
        .io_data                    (io_data),
        .dram_req                   (dram_req),
        .dram_req_write             (dram_req_write),
        .dram_req_read              (dram_req_read),
        .dram_req_addr              (dram_req_addr),
        .dram_req_size              (dram_req_size),
        .dram_req_data              (dram_req_data),
        .dram_expected_lat          (dram_expected_lat),
        .dram_fill_accept           (dram_fill_accept),
        .dram_fill_rsp              (dram_fill_rsp),
        .dram_fill_rsp_addr         (dram_fill_rsp_addr),
        .dram_fill_rsp_data         (dram_fill_rsp_data),
        .I_dram_req                 (I_dram_req),
        .I_dram_req_write           (I_dram_req_write),
        .I_dram_req_read            (I_dram_req_read),
        .I_dram_req_addr            (I_dram_req_addr),
        .I_dram_req_size            (I_dram_req_size),
        .I_dram_req_data            (I_dram_req_data),
        .I_dram_expected_lat        (I_dram_expected_lat),
        .I_dram_fill_accept         (I_dram_fill_accept),
        .I_dram_fill_rsp            (I_dram_fill_rsp),
        .I_dram_fill_rsp_addr       (I_dram_fill_rsp_addr),
        .I_dram_fill_rsp_data       (I_dram_fill_rsp_data),
        .out_ebreak                 (out_ebreak)
        );



	


	

endmodule