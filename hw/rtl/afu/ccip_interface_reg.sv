// Code reused from Intel OPAE's 04_local_memory sample program with changes made to fit Vortex

// Register all interface signals

import ccip_if_pkg::*;
module ccip_interface_reg(
  // CCI-P Clocks and Resets
  input           logic             pClk,                    // 400MHz - CC-P clock domain. Primary Clock
  input           logic             pck_cp2af_softReset_T0,  // CCI-P ACTIVE HIGH Soft Reset
  input           logic [1:0]       pck_cp2af_pwrState_T0,   // CCI-P AFU Power State
  input           logic             pck_cp2af_error_T0,      // CCI-P Protocol Error Detected
  // Interface structures
  input           t_if_ccip_Rx      pck_cp2af_sRx_T0,        // CCI-P Rx Port
  input           t_if_ccip_Tx      pck_af2cp_sTx_T0,        // CCI-P Tx Port
  
  output          logic             pck_cp2af_softReset_T1,
  output          logic [1:0]       pck_cp2af_pwrState_T1, 
  output          logic             pck_cp2af_error_T1,    
                                    
  output          t_if_ccip_Rx      pck_cp2af_sRx_T1,      
  output          t_if_ccip_Tx      pck_af2cp_sTx_T1

);
(* preserve *) logic             pck_cp2af_softReset_T0_q;
(* preserve *) logic [1:0]       pck_cp2af_pwrState_T0_q;
(* preserve *) logic             pck_cp2af_error_T0_q;
(* preserve *) t_if_ccip_Rx      pck_cp2af_sRx_T0_q;     
(* preserve *) t_if_ccip_Tx      pck_af2cp_sTx_T0_q;

always@(posedge pClk)
begin
    pck_cp2af_softReset_T0_q   <= pck_cp2af_softReset_T0;
    pck_cp2af_pwrState_T0_q    <= pck_cp2af_pwrState_T0;
    pck_cp2af_error_T0_q       <= pck_cp2af_error_T0;
    pck_cp2af_sRx_T0_q         <= pck_cp2af_sRx_T0;
    pck_af2cp_sTx_T0_q         <= pck_af2cp_sTx_T0;
end

always_comb
begin
    pck_cp2af_softReset_T1      = pck_cp2af_softReset_T0_q;
    pck_cp2af_pwrState_T1       = pck_cp2af_pwrState_T0_q;
    pck_cp2af_error_T1          = pck_cp2af_error_T0_q;
    pck_cp2af_sRx_T1            = pck_cp2af_sRx_T0_q;
    pck_af2cp_sTx_T1            = pck_af2cp_sTx_T0_q;
end

endmodule
