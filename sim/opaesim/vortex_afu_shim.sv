`include "VX_platform.vh"
`IGNORE_WARNINGS_BEGIN
`include "vortex_afu.vh"
`IGNORE_WARNINGS_END

/* verilator lint_off IMPORTSTAR */ 
import ccip_if_pkg::*;
import local_mem_cfg_pkg::*;
/* verilator lint_on IMPORTSTAR */

`include "VX_define.vh"

module vortex_afu_shim (
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

  input t_ccip_mmioAddr       vcp2af_sRxPort_c0_ReqMmioHdr_address,
  input logic [1:0]           vcp2af_sRxPort_c0_ReqMmioHdr_length, 
  input logic                 vcp2af_sRxPort_c0_ReqMmioHdr_rsvd,
  input t_ccip_tid            vcp2af_sRxPort_c0_ReqMmioHdr_tid, 
  
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
  output  t_local_mem_data      avs_writedata [`PLATFORM_PARAM_LOCAL_MEMORY_BANKS],
  input   t_local_mem_data      avs_readdata [`PLATFORM_PARAM_LOCAL_MEMORY_BANKS],
  output  t_local_mem_addr      avs_address [`PLATFORM_PARAM_LOCAL_MEMORY_BANKS],
  input   logic                 avs_waitrequest [`PLATFORM_PARAM_LOCAL_MEMORY_BANKS],
  output  logic                 avs_write [`PLATFORM_PARAM_LOCAL_MEMORY_BANKS],
  output  logic                 avs_read [`PLATFORM_PARAM_LOCAL_MEMORY_BANKS],
  output  t_local_mem_byte_mask avs_byteenable [`PLATFORM_PARAM_LOCAL_MEMORY_BANKS],
  output  t_local_mem_burst_cnt avs_burstcount [`PLATFORM_PARAM_LOCAL_MEMORY_BANKS],
  input                         avs_readdatavalid [`PLATFORM_PARAM_LOCAL_MEMORY_BANKS]
);

t_if_ccip_Rx cp2af_sRxPort;
t_if_ccip_Tx af2cp_sTxPort;

vortex_afu #(
  .NUM_LOCAL_MEM_BANKS(`PLATFORM_PARAM_LOCAL_MEMORY_BANKS)
) afu (
    .clk(clk),
    .reset(reset),
    .cp2af_sRxPort(cp2af_sRxPort),
    .af2cp_sTxPort(af2cp_sTxPort),
    .avs_writedata(avs_writedata),
    .avs_readdata(avs_readdata),
    .avs_address(avs_address),
    .avs_waitrequest(avs_waitrequest),
    .avs_write(avs_write),
    .avs_read(avs_read),
    .avs_byteenable(avs_byteenable),
    .avs_burstcount(avs_burstcount),
    .avs_readdatavalid(avs_readdatavalid)
);

t_if_ccip_c0_RxHdr c0_RxHdr;
always @ (*) begin
  c0_RxHdr = 'x;
  if (vcp2af_sRxPort_c0_mmioWrValid || vcp2af_sRxPort_c0_mmioRdValid) begin
    c0_RxHdr.reqMmioHdr.address = vcp2af_sRxPort_c0_ReqMmioHdr_address;
    c0_RxHdr.reqMmioHdr.length  = vcp2af_sRxPort_c0_ReqMmioHdr_length;
    c0_RxHdr.reqMmioHdr.rsvd    = vcp2af_sRxPort_c0_ReqMmioHdr_rsvd;
    c0_RxHdr.reqMmioHdr.tid     = vcp2af_sRxPort_c0_ReqMmioHdr_tid;   
  end else begin
    c0_RxHdr.rspMemHdr.vc_used  = vcp2af_sRxPort_c0_hdr_vc_used;
    c0_RxHdr.rspMemHdr.rsvd1    = vcp2af_sRxPort_c0_hdr_rsvd1;
    c0_RxHdr.rspMemHdr.hit_miss = vcp2af_sRxPort_c0_hdr_hit_miss;
    c0_RxHdr.rspMemHdr.rsvd0    = vcp2af_sRxPort_c0_hdr_rsvd0;
    c0_RxHdr.rspMemHdr.cl_num   = vcp2af_sRxPort_c0_hdr_cl_num;
    c0_RxHdr.rspMemHdr.resp_type = vcp2af_sRxPort_c0_hdr_resp_type;
    c0_RxHdr.rspMemHdr.mdata    = vcp2af_sRxPort_c0_hdr_mdata;
  end
end

assign cp2af_sRxPort.c0TxAlmFull = vcp2af_sRxPort_c0_TxAlmFull;
assign cp2af_sRxPort.c1TxAlmFull = vcp2af_sRxPort_c1_TxAlmFull;

assign cp2af_sRxPort.c0.hdr = c0_RxHdr;  
assign cp2af_sRxPort.c0.data = vcp2af_sRxPort_c0_data;
assign cp2af_sRxPort.c0.rspValid = vcp2af_sRxPort_c0_rspValid;
assign cp2af_sRxPort.c0.mmioRdValid = vcp2af_sRxPort_c0_mmioRdValid;
assign cp2af_sRxPort.c0.mmioWrValid = vcp2af_sRxPort_c0_mmioWrValid;

assign cp2af_sRxPort.c1.hdr.vc_used = vcp2af_sRxPort_c1_hdr_vc_used;
assign cp2af_sRxPort.c1.hdr.rsvd1 = vcp2af_sRxPort_c1_hdr_rsvd1;
assign cp2af_sRxPort.c1.hdr.hit_miss = vcp2af_sRxPort_c1_hdr_hit_miss;
assign cp2af_sRxPort.c1.hdr.format = vcp2af_sRxPort_c1_hdr_format;
assign cp2af_sRxPort.c1.hdr.rsvd0 = vcp2af_sRxPort_c1_hdr_rsvd0;
assign cp2af_sRxPort.c1.hdr.cl_num = vcp2af_sRxPort_c1_hdr_cl_num;
assign cp2af_sRxPort.c1.hdr.resp_type = vcp2af_sRxPort_c1_hdr_resp_type;
assign cp2af_sRxPort.c1.hdr.mdata = vcp2af_sRxPort_c1_hdr_mdata;    
assign cp2af_sRxPort.c1.rspValid = vcp2af_sRxPort_c1_rspValid;  

assign af2cp_sTxPort_c0_hdr_vc_sel = af2cp_sTxPort.c0.hdr.vc_sel;
assign af2cp_sTxPort_c0_hdr_rsvd1 = af2cp_sTxPort.c0.hdr.rsvd1;
assign af2cp_sTxPort_c0_hdr_cl_len = af2cp_sTxPort.c0.hdr.cl_len;
assign af2cp_sTxPort_c0_hdr_req_type = af2cp_sTxPort.c0.hdr.req_type;
assign af2cp_sTxPort_c0_hdr_rsvd0 = af2cp_sTxPort.c0.hdr.rsvd0;
assign af2cp_sTxPort_c0_hdr_address = af2cp_sTxPort.c0.hdr.address;
assign af2cp_sTxPort_c0_hdr_mdata = af2cp_sTxPort.c0.hdr.mdata;
assign af2cp_sTxPort_c0_valid = af2cp_sTxPort.c0.valid;

assign af2cp_sTxPort_c1_hdr_rsvd2 = af2cp_sTxPort.c1.hdr.rsvd2;
assign af2cp_sTxPort_c1_hdr_vc_sel = af2cp_sTxPort.c1.hdr.vc_sel;
assign af2cp_sTxPort_c1_hdr_sop = af2cp_sTxPort.c1.hdr.sop;
assign af2cp_sTxPort_c1_hdr_rsvd1 = af2cp_sTxPort.c1.hdr.rsvd1;
assign af2cp_sTxPort_c1_hdr_cl_len = af2cp_sTxPort.c1.hdr.cl_len;
assign af2cp_sTxPort_c1_hdr_req_type = af2cp_sTxPort.c1.hdr.req_type;
assign af2cp_sTxPort_c1_hdr_rsvd0 = af2cp_sTxPort.c1.hdr.rsvd0;
assign af2cp_sTxPort_c1_hdr_address = af2cp_sTxPort.c1.hdr.address;
assign af2cp_sTxPort_c1_hdr_mdata = af2cp_sTxPort.c1.hdr.mdata;
assign af2cp_sTxPort_c1_data = af2cp_sTxPort.c1.data;         
assign af2cp_sTxPort_c1_valid = af2cp_sTxPort.c1.valid;

assign af2cp_sTxPort_c2_hdr_tid = af2cp_sTxPort.c2.hdr.tid; 
assign af2cp_sTxPort_c2_mmioRdValid = af2cp_sTxPort.c2.mmioRdValid;  
assign af2cp_sTxPort_c2_data = af2cp_sTxPort.c2.data;

endmodule
