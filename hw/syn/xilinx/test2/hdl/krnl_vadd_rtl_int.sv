/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

///////////////////////////////////////////////////////////////////////////////
// Description: This is a example of how to create an RTL Kernel.  The function
// of this module is to add two 32-bit values and produce a result.  The values
// are read from one AXI4 memory mapped master, processed and then written out.
//
// Data flow: axi_read_master->fifo[2]->adder->fifo->axi_write_master
///////////////////////////////////////////////////////////////////////////////

// default_nettype of none prevents implicit wire declaration.
`default_nettype none
`timescale 1 ns / 1 ps 

module krnl_vadd_rtl_int #( 
  parameter integer  C_S_AXI_CTRL_DATA_WIDTH = 32,
  parameter integer  C_S_AXI_CTRL_ADDR_WIDTH = 6,
  parameter integer  C_M_AXI_MEM_ID_WIDTH = 1,
  parameter integer  C_M_AXI_MEM_ADDR_WIDTH = 32,
  parameter integer  C_M_AXI_MEM_DATA_WIDTH = 32
)
(
  // System signals
  input  wire  ap_clk,
  input  wire  ap_rst_n,

  // AXI4 master interface 
  output wire                                 m_axi_mem_awvalid,
  input  wire                                 m_axi_mem_awready,
  output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]    m_axi_mem_awaddr,
  output wire [C_M_AXI_MEM_ID_WIDTH - 1:0]    m_axi_mem_awid,
  output wire [7:0]                           m_axi_mem_awlen,
  output wire [2:0]                           m_axi_mem_awsize,
  output wire [1:0]                           m_axi_mem_awburst,
  output wire [1:0]                           m_axi_mem_awlock,
  output wire [3:0]                           m_axi_mem_awcache,
  output wire [2:0]                           m_axi_mem_awprot,
  output wire [3:0]                           m_axi_mem_awqos,
  output wire [3:0]                           m_axi_mem_awregion,

  output wire                                 m_axi_mem_wvalid,
  input  wire                                 m_axi_mem_wready,
  output wire [C_M_AXI_MEM_DATA_WIDTH-1:0]    m_axi_mem_wdata,
  output wire [C_M_AXI_MEM_DATA_WIDTH/8-1:0]  m_axi_mem_wstrb,
  output wire                                 m_axi_mem_wlast,
  
  input  wire                                 m_axi_mem_bvalid,
  output wire                                 m_axi_mem_bready,
  input  wire [1:0]                           m_axi_mem_bresp,
  input  wire [C_M_AXI_MEM_ID_WIDTH - 1:0]    m_axi_mem_bid,
  
  output wire                                 m_axi_mem_arvalid,
  input  wire                                 m_axi_mem_arready,
  output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]    m_axi_mem_araddr,
  output wire [C_M_AXI_MEM_ID_WIDTH-1:0]      m_axi_mem_arid,
  output wire [7:0]                           m_axi_mem_arlen,
  output wire [2:0]                           m_axi_mem_arsize,
  output wire [1:0]                           m_axi_mem_arburst,
  output wire [1:0]                           m_axi_mem_arlock,
  output wire [3:0]                           m_axi_mem_arcache,
  output wire [2:0]                           m_axi_mem_arprot,
  output wire [3:0]                           m_axi_mem_arqos,
  output wire [3:0]                           m_axi_mem_arregion,
  
  input  wire                                 m_axi_mem_rvalid,
  output wire                                 m_axi_mem_rready,
  input  wire [C_M_AXI_MEM_DATA_WIDTH - 1:0]  m_axi_mem_rdata,
  input  wire                                 m_axi_mem_rlast,
  input  wire [C_M_AXI_MEM_ID_WIDTH - 1:0]    m_axi_mem_rid,
  input  wire [1:0]                           m_axi_mem_rresp,

  // AXI4-Lite slave interface
  input  wire                                 s_axi_ctrl_awvalid,
  output wire                                 s_axi_ctrl_awready,
  input  wire [C_S_AXI_CTRL_ADDR_WIDTH-1:0]   s_axi_ctrl_awaddr,
  input  wire                                 s_axi_ctrl_wvalid,
  output wire                                 s_axi_ctrl_wready,
  input  wire [C_S_AXI_CTRL_DATA_WIDTH-1:0]   s_axi_ctrl_wdata,
  input  wire [C_S_AXI_CTRL_DATA_WIDTH/8-1:0] s_axi_ctrl_wstrb,
  input  wire                                 s_axi_ctrl_arvalid,
  output wire                                 s_axi_ctrl_arready,
  input  wire [C_S_AXI_CTRL_ADDR_WIDTH-1:0]   s_axi_ctrl_araddr,
  output wire                                 s_axi_ctrl_rvalid,
  input  wire                                 s_axi_ctrl_rready,
  output wire [C_S_AXI_CTRL_DATA_WIDTH-1:0]   s_axi_ctrl_rdata,
  output wire [1:0]                           s_axi_ctrl_rresp,
  output wire                                 s_axi_ctrl_bvalid,
  input  wire                                 s_axi_ctrl_bready,
  output wire [1:0]                           s_axi_ctrl_bresp,
  output wire                                 interrupt 
);
///////////////////////////////////////////////////////////////////////////////
// Local Parameters (constants)
///////////////////////////////////////////////////////////////////////////////
localparam integer LP_NUM_READ_CHANNELS  = 2;
localparam integer LP_LENGTH_WIDTH       = 32;
localparam integer LP_DW_BYTES           = C_M_AXI_MEM_DATA_WIDTH/8;
localparam integer LP_AXI_BURST_LEN      = 4096/LP_DW_BYTES < 256 ? 4096/LP_DW_BYTES : 256;
localparam integer LP_LOG_BURST_LEN      = $clog2(LP_AXI_BURST_LEN);
localparam integer LP_RD_MAX_OUTSTANDING = 3;
localparam integer LP_RD_FIFO_DEPTH      = LP_AXI_BURST_LEN*(LP_RD_MAX_OUTSTANDING + 1);
localparam integer LP_WR_FIFO_DEPTH      = LP_AXI_BURST_LEN;


///////////////////////////////////////////////////////////////////////////////
// Variables
///////////////////////////////////////////////////////////////////////////////
logic areset = 1'b0;  
logic ap_start;
logic ap_start_pulse;
logic ap_start_r;
logic ap_ready;
logic ap_done;
logic ap_idle = 1'b1;
logic [C_M_AXI_MEM_ADDR_WIDTH-1:0] a;
logic [C_M_AXI_MEM_ADDR_WIDTH-1:0] b;
logic [C_M_AXI_MEM_ADDR_WIDTH-1:0] c;
logic [LP_LENGTH_WIDTH-1:0]         length_r;

logic read_done;
logic [LP_NUM_READ_CHANNELS-1:0] rd_tvalid;
logic [LP_NUM_READ_CHANNELS-1:0] rd_tready_n; 
logic [LP_NUM_READ_CHANNELS-1:0] [C_M_AXI_MEM_DATA_WIDTH-1:0] rd_tdata;
logic [LP_NUM_READ_CHANNELS-1:0] ctrl_rd_fifo_prog_full;
logic [LP_NUM_READ_CHANNELS-1:0] rd_fifo_tvalid_n;
logic [LP_NUM_READ_CHANNELS-1:0] rd_fifo_tready; 
logic [LP_NUM_READ_CHANNELS-1:0] [C_M_AXI_MEM_DATA_WIDTH-1:0] rd_fifo_tdata;

logic                               adder_tvalid;
logic                               adder_tready_n; 
logic [C_M_AXI_MEM_DATA_WIDTH-1:0] adder_tdata;
logic                               wr_fifo_tvalid_n;
logic                               wr_fifo_tready; 
logic [C_M_AXI_MEM_DATA_WIDTH-1:0] wr_fifo_tdata;

///////////////////////////////////////////////////////////////////////////////
// RTL Logic 
///////////////////////////////////////////////////////////////////////////////
// Tie-off unused AXI protocol features
assign m_axi_mem_awid     = {C_M_AXI_MEM_ID_WIDTH{1'b0}};
assign m_axi_mem_awburst  = 2'b01;
assign m_axi_mem_awlock   = 2'b00;
assign m_axi_mem_awcache  = 4'b0000;
assign m_axi_mem_awprot   = 3'b000;
assign m_axi_mem_awqos    = 4'b0000;
assign m_axi_mem_awregion = 4'b0000;
assign m_axi_mem_arburst  = 2'b01;
assign m_axi_mem_arlock   = 2'b00;
assign m_axi_mem_arcache  = 4'b0000;
assign m_axi_mem_arprot   = 3'b000;
assign m_axi_mem_arqos    = 4'b0000;
assign m_axi_mem_arregion = 4'b0000;

// Register and invert reset signal for better timing.
always @(posedge ap_clk) begin 
  areset <= ~ap_rst_n; 
end

// create pulse when ap_start transitions to 1
always @(posedge ap_clk) begin 
  begin 
    ap_start_r <= ap_start;
  end
end

assign ap_start_pulse = ap_start & ~ap_start_r;

// ap_idle is asserted when done is asserted, it is de-asserted when ap_start_pulse 
// is asserted
always @(posedge ap_clk) begin 
  if (areset) begin 
    ap_idle <= 1'b1;
  end
  else begin 
    ap_idle <= ap_done        ? 1'b1 : 
               ap_start_pulse ? 1'b0 : 
                                ap_idle;
  end
end

assign ap_ready = ap_done;

// AXI4-Lite slave
krnl_vadd_rtl_control_s_axi #(
  .C_S_AXI_ADDR_WIDTH( C_S_AXI_CTRL_ADDR_WIDTH ),
  .C_S_AXI_DATA_WIDTH( C_S_AXI_CTRL_DATA_WIDTH )
) 
inst_krnl_vadd_control_s_axi (
  .AWVALID   ( s_axi_ctrl_awvalid         ) ,
  .AWREADY   ( s_axi_ctrl_awready         ) ,
  .AWADDR    ( s_axi_ctrl_awaddr          ) ,
  .WVALID    ( s_axi_ctrl_wvalid          ) ,
  .WREADY    ( s_axi_ctrl_wready          ) ,
  .WDATA     ( s_axi_ctrl_wdata           ) ,
  .WSTRB     ( s_axi_ctrl_wstrb           ) ,
  .ARVALID   ( s_axi_ctrl_arvalid         ) ,
  .ARREADY   ( s_axi_ctrl_arready         ) ,
  .ARADDR    ( s_axi_ctrl_araddr          ) ,
  .RVALID    ( s_axi_ctrl_rvalid          ) ,
  .RREADY    ( s_axi_ctrl_rready          ) ,
  .RDATA     ( s_axi_ctrl_rdata           ) ,
  .RRESP     ( s_axi_ctrl_rresp           ) ,
  .BVALID    ( s_axi_ctrl_bvalid          ) ,
  .BREADY    ( s_axi_ctrl_bready          ) ,
  .BRESP     ( s_axi_ctrl_bresp           ) ,
  .ACLK      ( ap_clk                        ) ,
  .ARESET    ( areset                        ) ,
  .ACLK_EN   ( 1'b1                          ) ,
  .ap_start  ( ap_start                      ) ,
  .interrupt ( interrupt                     ) ,
  .ap_ready  ( ap_ready                      ) ,
  .ap_done   ( ap_done                       ) ,
  .ap_idle   ( ap_idle                       ) ,
  .a         ( a[0+:C_M_AXI_MEM_ADDR_WIDTH] ) ,
  .b         ( b[0+:C_M_AXI_MEM_ADDR_WIDTH] ) ,
  .c         ( c[0+:C_M_AXI_MEM_ADDR_WIDTH] ) ,
  .length_r  ( length_r[0+:LP_LENGTH_WIDTH]  ) 
);

// AXI4 Read Master
krnl_vadd_rtl_axi_read_master #( 
  .C_ADDR_WIDTH       ( C_M_AXI_MEM_ADDR_WIDTH ) ,
  .C_DATA_WIDTH       ( C_M_AXI_MEM_DATA_WIDTH ) ,
  .C_ID_WIDTH         ( C_M_AXI_MEM_ID_WIDTH   ) ,
  .C_NUM_CHANNELS     ( LP_NUM_READ_CHANNELS    ) ,
  .C_LENGTH_WIDTH     ( LP_LENGTH_WIDTH         ) ,
  .C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
  .C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) ,
  .C_MAX_OUTSTANDING  ( LP_RD_MAX_OUTSTANDING   )
)
inst_axi_read_master ( 
  .aclk           ( ap_clk                 ) ,
  .areset         ( areset                 ) ,

  .ctrl_start     ( ap_start_pulse         ) ,
  .ctrl_done      ( read_done              ) ,
  .ctrl_offset    ( {b,a}                  ) ,
  .ctrl_length    ( length_r               ) ,
  .ctrl_prog_full ( ctrl_rd_fifo_prog_full ) ,

  .arvalid        ( m_axi_mem_arvalid     ) ,
  .arready        ( m_axi_mem_arready     ) ,
  .araddr         ( m_axi_mem_araddr      ) ,
  .arid           ( m_axi_mem_arid        ) ,
  .arlen          ( m_axi_mem_arlen       ) ,
  .arsize         ( m_axi_mem_arsize      ) ,
  .rvalid         ( m_axi_mem_rvalid      ) ,
  .rready         ( m_axi_mem_rready      ) ,
  .rdata          ( m_axi_mem_rdata       ) ,
  .rlast          ( m_axi_mem_rlast       ) ,
  .rid            ( m_axi_mem_rid         ) ,
  .rresp          ( m_axi_mem_rresp       ) ,

  .m_tvalid       ( rd_tvalid              ) ,
  .m_tready       ( ~rd_tready_n           ) ,
  .m_tdata        ( rd_tdata               ) 
);

// xpm_fifo_sync: Synchronous FIFO
// Xilinx Parameterized Macro, Version 2016.4
xpm_fifo_sync # (
  .FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
  .ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
  .FIFO_WRITE_DEPTH          (LP_RD_FIFO_DEPTH),   //positive integer
  .WRITE_DATA_WIDTH          (C_M_AXI_MEM_DATA_WIDTH),        //positive integer
  .WR_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),       //positive integer, Not used
  .PROG_FULL_THRESH          (LP_AXI_BURST_LEN-2),               //positive integer
  .FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
  .READ_MODE                 ("fwft"),            //string; "std" or "fwft";
  .FIFO_READ_LATENCY         (1),                //positive integer;
  .READ_DATA_WIDTH           (C_M_AXI_MEM_DATA_WIDTH),               //positive integer
  .RD_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),               //positive integer, not used
  .PROG_EMPTY_THRESH         (10),               //positive integer, not used 
  .DOUT_RESET_VALUE          ("0"),              //string, don't care
  .WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

) inst_rd_xpm_fifo_sync[LP_NUM_READ_CHANNELS-1:0] (
  .sleep         ( 1'b0             ) ,
  .rst           ( areset           ) ,
  .wr_clk        ( ap_clk           ) ,
  .wr_en         ( rd_tvalid        ) ,
  .din           ( rd_tdata         ) ,
  .full          ( rd_tready_n      ) ,
  .prog_full     ( ctrl_rd_fifo_prog_full) ,
  .wr_data_count (                  ) ,
  .overflow      (                  ) ,
  .wr_rst_busy   (                  ) ,
  .rd_en         ( rd_fifo_tready   ) ,
  .dout          ( rd_fifo_tdata    ) ,
  .empty         ( rd_fifo_tvalid_n ) ,
  .prog_empty    (                  ) ,
  .rd_data_count (                  ) ,
  .underflow     (                  ) ,
  .rd_rst_busy   (                  ) ,
  .injectsbiterr ( 1'b0             ) ,
  .injectdbiterr ( 1'b0             ) ,
  .sbiterr       (                  ) ,
  .dbiterr       (                  ) 

);

// Combinatorial Adder
krnl_vadd_rtl_adder #( 
  .C_DATA_WIDTH   ( C_M_AXI_MEM_DATA_WIDTH ) ,
  .C_NUM_CHANNELS ( LP_NUM_READ_CHANNELS    ) 
)
inst_adder ( 
  .aclk     ( ap_clk            ) ,
  .areset   ( areset            ) ,

  .s_tvalid ( ~rd_fifo_tvalid_n ) ,
  .s_tready ( rd_fifo_tready    ) ,
  .s_tdata  ( rd_fifo_tdata     ) ,

  .m_tvalid ( adder_tvalid      ) ,
  .m_tready ( ~adder_tready_n   ) ,
  .m_tdata  ( adder_tdata       ) 
);

// xpm_fifo_sync: Synchronous FIFO
// Xilinx Parameterized Macro, Version 2016.4
xpm_fifo_sync # (
  .FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
  .ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
  .FIFO_WRITE_DEPTH          (LP_WR_FIFO_DEPTH),   //positive integer
  .WRITE_DATA_WIDTH          (C_M_AXI_MEM_DATA_WIDTH),               //positive integer
  .WR_DATA_COUNT_WIDTH       ($clog2(LP_WR_FIFO_DEPTH)),               //positive integer, Not used
  .PROG_FULL_THRESH          (10),               //positive integer, Not used 
  .FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
  .READ_MODE                 ("fwft"),            //string; "std" or "fwft";
  .FIFO_READ_LATENCY         (1),                //positive integer;
  .READ_DATA_WIDTH           (C_M_AXI_MEM_DATA_WIDTH),               //positive integer
  .RD_DATA_COUNT_WIDTH       ($clog2(LP_WR_FIFO_DEPTH)),               //positive integer, not used
  .PROG_EMPTY_THRESH         (10),               //positive integer, not used 
  .DOUT_RESET_VALUE          ("0"),              //string, don't care
  .WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

) inst_wr_xpm_fifo_sync (
  .sleep         ( 1'b0             ) ,
  .rst           ( areset           ) ,
  .wr_clk        ( ap_clk           ) ,
  .wr_en         ( adder_tvalid     ) ,
  .din           ( adder_tdata      ) ,
  .full          ( adder_tready_n   ) ,
  .prog_full     (                  ) ,
  .wr_data_count (                  ) ,
  .overflow      (                  ) ,
  .wr_rst_busy   (                  ) ,
  .rd_en         ( wr_fifo_tready   ) ,
  .dout          ( wr_fifo_tdata    ) ,
  .empty         ( wr_fifo_tvalid_n ) ,
  .prog_empty    (                  ) ,
  .rd_data_count (                  ) ,
  .underflow     (                  ) ,
  .rd_rst_busy   (                  ) ,
  .injectsbiterr ( 1'b0             ) ,
  .injectdbiterr ( 1'b0             ) ,
  .sbiterr       (                  ) ,
  .dbiterr       (                  ) 

);


// AXI4 Write Master
krnl_vadd_rtl_axi_write_master #( 
  .C_ADDR_WIDTH       ( C_M_AXI_MEM_ADDR_WIDTH ) ,
  .C_DATA_WIDTH       ( C_M_AXI_MEM_DATA_WIDTH ) ,
  .C_MAX_LENGTH_WIDTH ( LP_LENGTH_WIDTH     ) ,
  .C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
  .C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) 
)
inst_axi_write_master ( 
  .aclk        ( ap_clk             ) ,
  .areset      ( areset             ) ,

  .ctrl_start  ( ap_start_pulse     ) ,
  .ctrl_offset ( c                  ) ,
  .ctrl_length ( length_r           ) ,
  .ctrl_done   ( ap_done            ) ,

  .awvalid     ( m_axi_mem_awvalid ) ,
  .awready     ( m_axi_mem_awready ) ,
  .awaddr      ( m_axi_mem_awaddr  ) ,
  .awlen       ( m_axi_mem_awlen   ) ,
  .awsize      ( m_axi_mem_awsize  ) ,

  .s_tvalid    ( ~wr_fifo_tvalid_n   ) ,
  .s_tready    ( wr_fifo_tready     ) ,
  .s_tdata     ( wr_fifo_tdata      ) ,

  .wvalid      ( m_axi_mem_wvalid  ) ,
  .wready      ( m_axi_mem_wready  ) ,
  .wdata       ( m_axi_mem_wdata   ) ,
  .wstrb       ( m_axi_mem_wstrb   ) ,
  .wlast       ( m_axi_mem_wlast   ) ,

  .bvalid      ( m_axi_mem_bvalid  ) ,
  .bready      ( m_axi_mem_bready  ) ,
  .bresp       ( m_axi_mem_bresp   ) 
);

endmodule : krnl_vadd_rtl_int

`default_nettype wire
