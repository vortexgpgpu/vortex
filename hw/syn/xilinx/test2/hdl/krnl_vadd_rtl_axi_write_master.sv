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

////////////////////////////////////////////////////////////
// Description: AXI4 Write Master.  Takes a stream of data in,
// appends address information and sends it out.
`default_nettype none

module krnl_vadd_rtl_axi_write_master #(
  parameter integer C_ADDR_WIDTH       = 64,
  parameter integer C_DATA_WIDTH       = 32,
  parameter integer C_MAX_LENGTH_WIDTH = 32,
  parameter integer C_BURST_LEN        = 256,
  parameter integer C_LOG_BURST_LEN    = 8
)
(
  // Control interface
  input wire                          ctrl_start,
  input wire [C_ADDR_WIDTH-1:0]       ctrl_offset,
  input wire [C_MAX_LENGTH_WIDTH-1:0] ctrl_length,
  output wire                         ctrl_done,

  // AXI4-Stream interface
  input wire                          s_tvalid,
  input wire  [C_DATA_WIDTH-1:0]      s_tdata,
  output wire                         s_tready,

  // AXI Interface 
  input wire                          aclk,
  input wire                          areset,

  output wire [C_ADDR_WIDTH-1:0]      awaddr,
  output wire [7:0]                   awlen,
  output wire [2:0]                   awsize,
  output wire                         awvalid,
  input wire                          awready,

  output wire [C_DATA_WIDTH-1:0]      wdata,
  output wire [C_DATA_WIDTH/8-1:0]    wstrb,
  output wire                         wlast,
  output wire                         wvalid,
  input wire                          wready,

  input wire [1:0]                    bresp,
  input wire                          bvalid,
  output wire                         bready
);

timeunit 1ps; 
timeprecision 1ps; 

/////////////////////////////////////////////////////////////////////////////
// Local Parameters
/////////////////////////////////////////////////////////////////////////////
localparam integer LP_LOG_MAX_W_TO_AW = 8; // Allow up to 256 outstanding w to aw transactions
localparam integer LP_TRANSACTION_CNTR_WIDTH = C_MAX_LENGTH_WIDTH-C_LOG_BURST_LEN;

/////////////////////////////////////////////////////////////////////////////
// Variables
/////////////////////////////////////////////////////////////////////////////
logic [LP_TRANSACTION_CNTR_WIDTH-1:0] num_full_bursts;
logic                                 num_partial_bursts;
logic                                 start    = 1'b0;
logic [LP_TRANSACTION_CNTR_WIDTH-1:0] num_transactions;
logic                                 has_partial_burst;
logic [C_LOG_BURST_LEN-1:0]           final_burst_len;
logic                                 single_transaction;

logic                                 wxfer;       // Unregistered write data transfer
logic                                 wfirst = 1'b1;
logic                                 load_burst_cntr;
logic [C_LOG_BURST_LEN-1:0]           wxfers_to_go;  // Used for simulation debug
logic [LP_TRANSACTION_CNTR_WIDTH-1:0] w_transactions_to_go;
logic                                 w_final_transaction;
logic                                 w_almost_final_transaction = 1'b0;

logic                                 awxfer;
logic                                 awvalid_r    = 1'b0; 
logic [C_ADDR_WIDTH-1:0]              addr;
logic                                 wfirst_d1    = 1'b0; 
logic                                 wfirst_pulse = 1'b0;
logic [LP_LOG_MAX_W_TO_AW-1:0]        dbg_w_to_aw_outstanding;
logic                                 idle_aw;
logic [LP_TRANSACTION_CNTR_WIDTH-1:0] aw_transactions_to_go;
logic                                 aw_final_transaction;

wire                                  bxfer;
logic [LP_TRANSACTION_CNTR_WIDTH-1:0] b_transactions_to_go;
logic                                 b_final_transaction;

/////////////////////////////////////////////////////////////////////////////
// Control logic
/////////////////////////////////////////////////////////////////////////////
// Count the number of transfers and assert done when the last bvalid is received.
assign num_full_bursts = ctrl_length[C_LOG_BURST_LEN+:C_MAX_LENGTH_WIDTH-C_LOG_BURST_LEN];
assign num_partial_bursts = ctrl_length[0+:C_LOG_BURST_LEN] ? 1'b1 : 1'b0; 

always @(posedge aclk) begin 
  start <= ctrl_start;
  num_transactions <= (num_partial_bursts == 1'b0) ? num_full_bursts - 1'b1 : num_full_bursts;
  has_partial_burst <= num_partial_bursts;
  final_burst_len <=  ctrl_length[0+:C_LOG_BURST_LEN] - 1'b1;
end

assign ctrl_done = bxfer & b_final_transaction;
assign single_transaction = (num_transactions == {LP_TRANSACTION_CNTR_WIDTH{1'b0}}) ? 1'b1 : 1'b0;

/////////////////////////////////////////////////////////////////////////////
// AXI Write Data Channel
/////////////////////////////////////////////////////////////////////////////
assign wvalid        = s_tvalid;
assign wdata         = s_tdata;
assign wstrb         = {(C_DATA_WIDTH/8){1'b1}};
assign s_tready = wready;

assign wxfer = wvalid & wready; 

always @(posedge aclk) begin 
  if (areset) begin 
    wfirst <= 1'b1;
  end
  else begin
    wfirst <= wxfer ? wlast : wfirst;
  end
end

// Load burst counter with partial burst if on final transaction or if there is only 1 transaction
assign load_burst_cntr = (wxfer & wlast & w_almost_final_transaction) || (start & single_transaction);

krnl_vadd_rtl_counter #(
  .C_WIDTH ( C_LOG_BURST_LEN         ) ,
  .C_INIT  ( {C_LOG_BURST_LEN{1'b1}} ) 
)
inst_burst_cntr ( 
  .clk        ( aclk            ) ,
  .clken      ( 1'b1            ) ,
  .rst        ( areset          ) ,
  .load       ( load_burst_cntr ) ,
  .incr       ( 1'b0            ) ,
  .decr       ( wxfer           ) ,
  .load_value ( final_burst_len ) ,
  .count      ( wxfers_to_go    ) ,
  .is_zero    ( wlast           ) 
);

krnl_vadd_rtl_counter #(
  .C_WIDTH ( LP_TRANSACTION_CNTR_WIDTH         ) ,
  .C_INIT  ( {LP_TRANSACTION_CNTR_WIDTH{1'b0}} ) 
)
inst_w_transaction_cntr ( 
  .clk        ( aclk                 ) ,
  .clken      ( 1'b1                 ) ,
  .rst        ( areset               ) ,
  .load       ( start                ) ,
  .incr       ( 1'b0                 ) ,
  .decr       ( wxfer & wlast        ) ,
  .load_value ( num_transactions     ) ,
  .count      ( w_transactions_to_go ) ,
  .is_zero    ( w_final_transaction  ) 
);

always @(posedge aclk) begin
  w_almost_final_transaction <= (w_transactions_to_go == 1) ? 1'b1 : 1'b0;
end

/////////////////////////////////////////////////////////////////////////////
// AXI Write Address Channel
/////////////////////////////////////////////////////////////////////////////
// The address channel samples the data channel and send out transactions when 
// first beat of wdata is asserted. This ensures that address requests are not 
// sent without data on the way.

assign awvalid = awvalid_r;
assign awxfer = awvalid & awready;

always @(posedge aclk) begin 
  if (areset) begin 
    awvalid_r <= 1'b0;
  end
  else begin
    awvalid_r <= ~idle_aw & ~awvalid_r ? 1'b1 : 
                 awready               ? 1'b0 : 
                                         awvalid_r;
  end
end

assign awaddr = addr;

always @(posedge aclk) begin 
  addr <= ctrl_start ? ctrl_offset :
          awxfer     ? addr + C_BURST_LEN*C_DATA_WIDTH/8 : 
                       addr;
end

assign awlen   = aw_final_transaction || (start & single_transaction) ? final_burst_len : C_BURST_LEN - 1;
assign awsize  = $clog2((C_DATA_WIDTH/8));

krnl_vadd_rtl_counter #(
  .C_WIDTH (LP_LOG_MAX_W_TO_AW),
  .C_INIT ({LP_LOG_MAX_W_TO_AW{1'b0}})
)
inst_w_to_aw_cntr ( 
  .clk        ( aclk                    ) ,
  .clken      ( 1'b1                    ) ,
  .rst        ( areset                  ) ,
  .load       ( 1'b0                    ) ,
  .incr       ( wfirst_pulse            ) ,
  .decr       ( awxfer                  ) ,
  .load_value (                         ) ,
  .count      ( dbg_w_to_aw_outstanding ) ,
  .is_zero    ( idle_aw                 ) 
);

always @(posedge aclk) begin 
  wfirst_d1 <= wvalid & wfirst; 
end

always @(posedge aclk) begin 
  wfirst_pulse <= wvalid & wfirst & ~wfirst_d1;
end

krnl_vadd_rtl_counter #(
  .C_WIDTH ( LP_TRANSACTION_CNTR_WIDTH         ) ,
  .C_INIT  ( {LP_TRANSACTION_CNTR_WIDTH{1'b0}} ) 
)
inst_aw_transaction_cntr ( 
  .clk        ( aclk                   ) ,
  .clken      ( 1'b1                   ) ,
  .rst        ( areset                 ) ,
  .load       ( start                  ) ,
  .incr       ( 1'b0                   ) ,
  .decr       ( awxfer                 ) ,
  .load_value ( num_transactions       ) ,
  .count      ( aw_transactions_to_go  ) ,
  .is_zero    ( aw_final_transaction   ) 
);

/////////////////////////////////////////////////////////////////////////////
// AXI Write Response Channel
/////////////////////////////////////////////////////////////////////////////

assign bready = 1'b1;
assign bxfer = bready & bvalid;

krnl_vadd_rtl_counter #(
  .C_WIDTH ( LP_TRANSACTION_CNTR_WIDTH         ) ,
  .C_INIT  ( {LP_TRANSACTION_CNTR_WIDTH{1'b0}} ) 
)
inst_b_transaction_cntr ( 
  .clk        ( aclk                 ) ,
  .clken      ( 1'b1                 ) ,
  .rst        ( areset               ) ,
  .load       ( start                ) ,
  .incr       ( 1'b0                 ) ,
  .decr       ( bxfer                ) ,
  .load_value ( num_transactions     ) ,
  .count      ( b_transactions_to_go ) ,
  .is_zero    ( b_final_transaction  ) 
);

endmodule : krnl_vadd_rtl_axi_write_master

`default_nettype wire
