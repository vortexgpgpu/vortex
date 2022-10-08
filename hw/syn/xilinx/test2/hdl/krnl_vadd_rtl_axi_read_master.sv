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
// Description: This is a multi-threaded AXI4 read master.  Each channel will
// issue commands on a different IDs.  As a result data may arrive out of 
// order.  The amount of data requested is equal to the ctrl_length variable.
// Prog full is set and sampled such that the FIFO will never overflow.  Thus 
// rready can be always asserted for better timing.
///////////////////////////////////////////////////////////////////////////////

`default_nettype none

module krnl_vadd_rtl_axi_read_master #( 
  parameter integer C_ID_WIDTH         = 1,   // Must be >= $clog2(C_NUM_CHANNELS)
  parameter integer C_ADDR_WIDTH       = 64,
  parameter integer C_DATA_WIDTH       = 32,
  parameter integer C_NUM_CHANNELS     = 2,   // Only 2 tested.
  parameter integer C_LENGTH_WIDTH     = 32,  
  parameter integer C_BURST_LEN        = 256, // Max AXI burst length for read commands
  parameter integer C_LOG_BURST_LEN    = 8,
  parameter integer C_MAX_OUTSTANDING  = 3 
)
(
  // System signals
  input  wire                                          aclk,
  input  wire                                          areset,
  // Control signals 
  input  wire                                          ctrl_start, 
  output wire                                          ctrl_done, 
  input  wire [C_NUM_CHANNELS-1:0][C_ADDR_WIDTH-1:0]   ctrl_offset,
  input  wire                     [C_LENGTH_WIDTH-1:0] ctrl_length,
  input  wire [C_NUM_CHANNELS-1:0]                     ctrl_prog_full,
  // AXI4 master interface                             
  output wire                                          arvalid,
  input  wire                                          arready,
  output wire [C_ADDR_WIDTH-1:0]                       araddr,
  output wire [C_ID_WIDTH-1:0]                         arid,
  output wire [7:0]                                    arlen,
  output wire [2:0]                                    arsize,
  input  wire                                          rvalid,
  output wire                                          rready,
  input  wire [C_DATA_WIDTH - 1:0]                     rdata,
  input  wire                                          rlast,
  input  wire [C_ID_WIDTH - 1:0]                       rid,
  input  wire [1:0]                                    rresp,
  // AXI4-Stream master interface, 1 interface per channel.
  output wire [C_NUM_CHANNELS-1:0]                     m_tvalid,
  input  wire [C_NUM_CHANNELS-1:0]                     m_tready,
  output wire [C_NUM_CHANNELS-1:0][C_DATA_WIDTH-1:0]   m_tdata
);

timeunit 1ps; 
timeprecision 1ps; 

///////////////////////////////////////////////////////////////////////////////
// Local Parameters
///////////////////////////////////////////////////////////////////////////////
localparam integer LP_MAX_OUTSTANDING_CNTR_WIDTH = $clog2(C_MAX_OUTSTANDING+1); 
localparam integer LP_TRANSACTION_CNTR_WIDTH = C_LENGTH_WIDTH-C_LOG_BURST_LEN;

///////////////////////////////////////////////////////////////////////////////
// Variables
///////////////////////////////////////////////////////////////////////////////
// Control logic
logic [C_NUM_CHANNELS-1:0]            done = '0;
logic [LP_TRANSACTION_CNTR_WIDTH-1:0] num_full_bursts;
logic                                 num_partial_bursts;
logic                                 start    = 1'b0;
logic [LP_TRANSACTION_CNTR_WIDTH-1:0] num_transactions;
logic                                 has_partial_burst;
logic [C_LOG_BURST_LEN-1:0]           final_burst_len;
logic                                 single_transaction;
logic                                 ar_idle = 1'b1;
logic                                 ar_done;
// AXI Read Address Channel
logic                                                         fifo_stall;
logic                                                         arxfer;
logic                                                         arvalid_r = 1'b0; 
logic [C_NUM_CHANNELS-1:0][C_ADDR_WIDTH-1:0]                  addr;
logic [C_ID_WIDTH-1:0]                                        id = {C_ID_WIDTH{1'b1}};
logic [LP_TRANSACTION_CNTR_WIDTH-1:0]                         ar_transactions_to_go;
logic                                                         ar_final_transaction;
logic [C_NUM_CHANNELS-1:0]                                    incr_ar_to_r_cnt;
logic [C_NUM_CHANNELS-1:0]                                    decr_ar_to_r_cnt;
logic [C_NUM_CHANNELS-1:0]                                    stall_ar;
logic [C_NUM_CHANNELS-1:0][LP_MAX_OUTSTANDING_CNTR_WIDTH-1:0] outstanding_vacancy_count;
// AXI Data Channel
logic [C_NUM_CHANNELS-1:0]                                tvalid;
logic [C_NUM_CHANNELS-1:0][C_DATA_WIDTH-1:0]              tdata;
logic                                                     rxfer;
logic [C_NUM_CHANNELS-1:0]                                decr_r_transaction_cntr;
logic [C_NUM_CHANNELS-1:0][LP_TRANSACTION_CNTR_WIDTH-1:0] r_transactions_to_go;
logic [C_NUM_CHANNELS-1:0]                                r_final_transaction;
///////////////////////////////////////////////////////////////////////////////
// Control Logic 
///////////////////////////////////////////////////////////////////////////////

always @(posedge aclk) begin
  for (int i = 0; i < C_NUM_CHANNELS; i++) begin 
    done[i] <= rxfer & rlast & (rid == i) & r_final_transaction[i] ? 1'b1 : 
          ctrl_done ? 1'b0 : done[i]; 
  end
end
assign ctrl_done = &done;

// Determine how many full burst to issue and if there are any partial bursts.
assign num_full_bursts = ctrl_length[C_LOG_BURST_LEN+:C_LENGTH_WIDTH-C_LOG_BURST_LEN];
assign num_partial_bursts = ctrl_length[0+:C_LOG_BURST_LEN] ? 1'b1 : 1'b0; 

always @(posedge aclk) begin 
  start <= ctrl_start;
  num_transactions <= (num_partial_bursts == 1'b0) ? num_full_bursts - 1'b1 : num_full_bursts;
  has_partial_burst <= num_partial_bursts;
  final_burst_len <=  ctrl_length[0+:C_LOG_BURST_LEN] - 1'b1;
end

// Special case if there is only 1 AXI transaction. 
assign single_transaction = (num_transactions == {LP_TRANSACTION_CNTR_WIDTH{1'b0}}) ? 1'b1 : 1'b0;

///////////////////////////////////////////////////////////////////////////////
// AXI Read Address Channel
///////////////////////////////////////////////////////////////////////////////
assign arvalid = arvalid_r;
assign araddr = addr[id];
assign arlen  = ar_final_transaction || (start & single_transaction) ? final_burst_len : C_BURST_LEN - 1;
assign arsize = $clog2((C_DATA_WIDTH/8));
assign arid   = id;

assign arxfer = arvalid & arready;
assign fifo_stall = ctrl_prog_full[id];

always @(posedge aclk) begin 
  if (areset) begin 
    arvalid_r <= 1'b0;
  end
  else begin
    arvalid_r <= ~ar_idle & ~stall_ar[id] & ~arvalid_r & ~fifo_stall ? 1'b1 : 
                 arready ? 1'b0 : arvalid_r;
  end
end

// When ar_idle, there are no transactions to issue.
always @(posedge aclk) begin 
  if (areset) begin 
    ar_idle <= 1'b1; 
  end
  else begin 
    ar_idle <= start   ? 1'b0 :
               ar_done ? 1'b1 : 
                         ar_idle;
  end
end

// each channel is assigned a different id. The transactions are interleaved.
always @(posedge aclk) begin 
  if (start) begin 
    id <= {C_ID_WIDTH{1'b1}};
  end
  else begin
    id <= arxfer ? id - 1'b1 : id; 
  end
end


// Increment to next address after each transaction is issued.
always @(posedge aclk) begin 
  for (int i = 0; i < C_NUM_CHANNELS; i++) begin
    addr[i] <= ctrl_start          ? ctrl_offset[i] :
               arxfer && (id == i) ? addr[i] + C_BURST_LEN*C_DATA_WIDTH/8 : 
                                     addr[i];
  end
end

// Counts down the number of transactions to send.
krnl_vadd_rtl_counter #(
  .C_WIDTH ( LP_TRANSACTION_CNTR_WIDTH         ) ,
  .C_INIT  ( {LP_TRANSACTION_CNTR_WIDTH{1'b0}} ) 
)
inst_ar_transaction_cntr ( 
  .clk        ( aclk                   ) ,
  .clken      ( 1'b1                   ) ,
  .rst        ( areset                 ) ,
  .load       ( start                  ) ,
  .incr       ( 1'b0                   ) ,
  .decr       ( arxfer && id == '0     ) ,
  .load_value ( num_transactions       ) ,
  .count      ( ar_transactions_to_go  ) ,
  .is_zero    ( ar_final_transaction   ) 
);

assign ar_done = ar_final_transaction && arxfer && id == 1'b0;

always_comb begin 
  for (int i = 0; i < C_NUM_CHANNELS; i++) begin 
    incr_ar_to_r_cnt[i] = rxfer & rlast & (rid == i);
    decr_ar_to_r_cnt[i] = arxfer & (arid == i);
  end
end

// Keeps track of the number of outstanding transactions. Stalls 
// when the value is reached so that the FIFO won't overflow.
krnl_vadd_rtl_counter #(
  .C_WIDTH ( LP_MAX_OUTSTANDING_CNTR_WIDTH                       ) ,
  .C_INIT  ( C_MAX_OUTSTANDING[0+:LP_MAX_OUTSTANDING_CNTR_WIDTH] ) 
)
inst_ar_to_r_transaction_cntr[C_NUM_CHANNELS-1:0] ( 
  .clk        ( aclk                           ) ,
  .clken      ( 1'b1                           ) ,
  .rst        ( areset                         ) ,
  .load       ( 1'b0                           ) ,
  .incr       ( incr_ar_to_r_cnt               ) ,
  .decr       ( decr_ar_to_r_cnt               ) ,
  .load_value ( {LP_MAX_OUTSTANDING_CNTR_WIDTH{1'b0}} ) ,
  .count      ( outstanding_vacancy_count      ) ,
  .is_zero    ( stall_ar                       ) 
);

///////////////////////////////////////////////////////////////////////////////
// AXI Read Channel
///////////////////////////////////////////////////////////////////////////////
assign m_tvalid = tvalid;
assign m_tdata = tdata;

always_comb begin 
  for (int i = 0; i < C_NUM_CHANNELS; i++) begin
    tvalid[i] = rvalid && (rid == i); 
    tdata[i] = rdata;
  end
end

// rready can remain high for optimal timing because ar transactions are not issued
// unless there is enough space in the FIFO.
assign rready = 1'b1;
assign rxfer = rready & rvalid;

always_comb begin 
  for (int i = 0; i < C_NUM_CHANNELS; i++) begin 
    decr_r_transaction_cntr[i] = rxfer & rlast & (rid == i);
  end
end
krnl_vadd_rtl_counter #(
  .C_WIDTH ( LP_TRANSACTION_CNTR_WIDTH         ) ,
  .C_INIT  ( {LP_TRANSACTION_CNTR_WIDTH{1'b0}} ) 
)
inst_r_transaction_cntr[C_NUM_CHANNELS-1:0] ( 
  .clk        ( aclk                          ) ,
  .clken      ( 1'b1                          ) ,
  .rst        ( areset                        ) ,
  .load       ( start                         ) ,
  .incr       ( 1'b0                          ) ,
  .decr       ( decr_r_transaction_cntr       ) ,
  .load_value ( num_transactions              ) ,
  .count      ( r_transactions_to_go          ) ,
  .is_zero    ( r_final_transaction           ) 
);


endmodule : krnl_vadd_rtl_axi_read_master

`default_nettype wire
