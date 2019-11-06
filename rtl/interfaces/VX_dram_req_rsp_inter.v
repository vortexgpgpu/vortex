
`include "../VX_define.v"

`ifndef VX_DRAM_REQ_RSP_INTER

`define VX_DRAM_REQ_RSP_INTER

interface VX_dram_req_rsp_inter ();

    // Req
    wire [31:0]                                                             o_m_evict_addr;
    wire [31:0]                                                             o_m_read_addr;
    wire                                                                    o_m_valid;
    wire[`DCACHE_NUMBER_BANKS - 1:0][`DCACHE_NUM_WORDS_PER_BLOCK-1:0][31:0] o_m_writedata;
    wire                                                                    o_m_read_or_write;

    // Rsp
    wire[`DCACHE_NUMBER_BANKS - 1:0][`DCACHE_NUM_WORDS_PER_BLOCK-1:0][31:0] i_m_readdata;
    wire                                                                    i_m_ready;


endinterface


`endif
