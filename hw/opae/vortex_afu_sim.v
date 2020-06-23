`include "vortex_afu.vh"

/* verilator lint_off IMPORTSTAR */ 
import ccip_if_pkg::*;
import local_mem_cfg_pkg::*;
/* verilator lint_on IMPORTSTAR */ 

module vortex_afu_sim #(
  parameter NUM_LOCAL_MEM_BANKS = 2
) (
  // global signals
  input clk,
  input reset,

  // IF signals between CCI and AFU
  input logic                 vcp2af_sRxPort_c0_TxAlmFull,  
  input logic                 vcp2af_sRxPort_c1_TxAlmFull,

  input t_ccip_vc             vcp2af_sRxPort_c0_hdr_vc_used,
  input logic                 vcp2af_sRxPort_c0_hdr_rsvd1,
  input logic                 vcp2af_sRxPort_c0_hdr_hit_miss,
  input logic [1:0]           vcp2af_sRxPort_c0_hdr_rsvd0,
  input t_ccip_clNum          vcp2af_sRxPort_c0_hdr_cl_num,
  input t_ccip_c0_rsp         vcp2af_sRxPort_c0_hdr_resp_type,
  input t_ccip_mdata          vcp2af_sRxPort_c0_hdr_mdata,
  input t_ccip_clData         vcp2af_sRxPort_c0_data,
  input logic                 vcp2af_sRxPort_c0_rspValid,       
  input logic                 vcp2af_sRxPort_c0_mmioRdValid,    
  input logic                 vcp2af_sRxPort_c0_mmioWrValid,    
  
  input t_ccip_vc             vcp2af_sRxPort_c1_hdr_vc_used,
  input logic                 vcp2af_sRxPort_c1_hdr_rsvd1,
  input logic                 vcp2af_sRxPort_c1_hdr_hit_miss,
  input logic                 vcp2af_sRxPort_c1_hdr_format,
  input logic                 vcp2af_sRxPort_c1_hdr_rsvd0,
  input t_ccip_clNum          vcp2af_sRxPort_c1_hdr_cl_num,
  input t_ccip_c1_rsp         vcp2af_sRxPort_c1_hdr_resp_type,
  input t_ccip_mdata          vcp2af_sRxPort_c1_hdr_mdata,     
  input logic                 vcp2af_sRxPort_c1_rspValid, 
  
  output t_ccip_vc            af2cp_sTxPort_c0_hdr_vc_sel,
  output logic [1:0]          af2cp_sTxPort_c0_hdr_rsvd1,    
  output t_ccip_clLen         af2cp_sTxPort_c0_hdr_cl_len,
  output t_ccip_c0_req        af2cp_sTxPort_c0_hdr_req_type,
  output logic [5:0]          af2cp_sTxPort_c0_hdr_rsvd0,     
  output t_ccip_clAddr        af2cp_sTxPort_c0_hdr_address,
  output t_ccip_mdata         af2cp_sTxPort_c0_hdr_mdata,
  output logic                af2cp_sTxPort_c0_valid,      

  output logic [5:0]          af2cp_sTxPort_c1_hdr_rsvd2,
  output t_ccip_vc            af2cp_sTxPort_c1_hdr_vc_sel,
  output logic                af2cp_sTxPort_c1_hdr_sop,
  output logic                af2cp_sTxPort_c1_hdr_rsvd1,     
  output t_ccip_clLen         af2cp_sTxPort_c1_hdr_cl_len,
  output t_ccip_c1_req        af2cp_sTxPort_c1_hdr_req_type,
  output logic [5:0]          af2cp_sTxPort_c1_hdr_rsvd0,     
  output t_ccip_clAddr        af2cp_sTxPort_c1_hdr_address,
  output t_ccip_mdata         af2cp_sTxPort_c1_hdr_mdata,
  output t_ccip_clData        af2cp_sTxPort_c1_data,          
  output logic                af2cp_sTxPort_c1_valid,         

  output t_ccip_tid           af2cp_sTxPort_c2_hdr_tid,
  output logic                af2cp_sTxPort_c2_mmioRdValid,   
  output t_ccip_mmioData      af2cp_sTxPort_c2_data,       
  
  // Avalon signals for local memory access
  output  t_local_mem_data      avs_writedata,
  input   t_local_mem_data      avs_readdata,
  output  t_local_mem_addr      avs_address,
  input   logic                 avs_waitrequest,
  output  logic                 avs_write,
  output  logic                 avs_read,
  output  t_local_mem_byte_mask avs_byteenable,
  output  t_local_mem_burst_cnt avs_burstcount,
  input                         avs_readdatavalid,

  output logic [$clog2(NUM_LOCAL_MEM_BANKS)-1:0] mem_bank_select
);

vortex_afu #(
  .NUM_LOCAL_MEM_BANKS(NUM_LOCAL_MEM_BANKS)
) vortex_afu (
    .clk(clk),
    .SoftReset(reset),
    .cp2af_sRxPort({
      vcp2af_sRxPort_c0_TxAlmFull,  
      vcp2af_sRxPort_c1_TxAlmFull,

      vcp2af_sRxPort_c0_hdr_vc_used,
      vcp2af_sRxPort_c0_hdr_rsvd1,
      vcp2af_sRxPort_c0_hdr_hit_miss,
      vcp2af_sRxPort_c0_hdr_rsvd0,
      vcp2af_sRxPort_c0_hdr_cl_num,
      vcp2af_sRxPort_c0_hdr_resp_type,
      vcp2af_sRxPort_c0_hdr_mdata,
      vcp2af_sRxPort_c0_data,
      vcp2af_sRxPort_c0_rspValid,       
      vcp2af_sRxPort_c0_mmioRdValid,    
      vcp2af_sRxPort_c0_mmioWrValid,    

      vcp2af_sRxPort_c1_hdr_vc_used,
      vcp2af_sRxPort_c1_hdr_rsvd1,
      vcp2af_sRxPort_c1_hdr_hit_miss,
      vcp2af_sRxPort_c1_hdr_format,
      vcp2af_sRxPort_c1_hdr_rsvd0,
      vcp2af_sRxPort_c1_hdr_cl_num,
      vcp2af_sRxPort_c1_hdr_resp_type,
      vcp2af_sRxPort_c1_hdr_mdata,     
      vcp2af_sRxPort_c1_rspValid}
    ),
    .af2cp_sTxPort({
      af2cp_sTxPort_c0_hdr_vc_sel,
      af2cp_sTxPort_c0_hdr_rsvd1,    
      af2cp_sTxPort_c0_hdr_cl_len,
      af2cp_sTxPort_c0_hdr_req_type,
      af2cp_sTxPort_c0_hdr_rsvd0,     
      af2cp_sTxPort_c0_hdr_address,
      af2cp_sTxPort_c0_hdr_mdata,
      af2cp_sTxPort_c0_valid,

      af2cp_sTxPort_c1_hdr_rsvd2,
      af2cp_sTxPort_c1_hdr_vc_sel,
      af2cp_sTxPort_c1_hdr_sop,
      af2cp_sTxPort_c1_hdr_rsvd1,     
      af2cp_sTxPort_c1_hdr_cl_len,
      af2cp_sTxPort_c1_hdr_req_type,
      af2cp_sTxPort_c1_hdr_rsvd0,     
      af2cp_sTxPort_c1_hdr_address,
      af2cp_sTxPort_c1_hdr_mdata,
      af2cp_sTxPort_c1_data,          
      af2cp_sTxPort_c1_valid,         

      af2cp_sTxPort_c2_hdr_tid,  
      af2cp_sTxPort_c2_mmioRdValid,   
      af2cp_sTxPort_c2_data
    }),
    .avs_writedata(avs_writedata),
    .avs_readdata(avs_readdata),
    .avs_address(avs_address),
    .avs_waitrequest(avs_waitrequest),
    .avs_write(avs_write),
    .avs_read(avs_read),
    .avs_byteenable(avs_byteenable),
    .avs_burstcount(avs_burstcount),
    .avs_readdatavalid(avs_readdatavalid),
    .mem_bank_select(mem_bank_select)
);

endmodule