// Code reused from Intel OPAE's 04_local_memory sample program with changes made to fit Vortex

// Top Level Vortex Driver

// To be done:
// Check how to run this with OPAE. Looks like setup issue


`include "platform_if.vh"

import local_mem_cfg_pkg::*;

module ccip_std_afu #(
    parameter NUM_LOCAL_MEM_BANKS = 2
) (
    // CCI-P Clocks and Resets
    input  logic        pClk,                 // Primary CCI-P interface clock.
    input  logic        pClkDiv2,             // Aligned, pClk divided by 2.
    input  logic        pClkDiv4,             // Aligned, pClk divided by 4.
    input  logic        uClk_usr,             // User clock domain. Refer to clock programming guide.
    input  logic        uClk_usrDiv2,         // Aligned, user clock divided by 2.
    input  logic        pck_cp2af_softReset,  // CCI-P ACTIVE HIGH Soft Reset

    input  logic [1:0]  pck_cp2af_pwrState,   // CCI-P AFU Power State
    input  logic        pck_cp2af_error,      // CCI-P Protocol Error Detected

    // CCI-P structures
    input  t_if_ccip_Rx pck_cp2af_sRx,        // CCI-P Rx Port
    output t_if_ccip_Tx pck_af2cp_sTx,        // CCI-P Tx Port

    // Local memory interface
    avalon_mem_if.to_fiu local_mem[NUM_LOCAL_MEM_BANKS]
);

    // ====================================================================
    // Pick the proper clk and reset, as chosen by the AFU's JSON file
    // ====================================================================

    // The platform may transform the CCI-P clock from pClk to a clock
    // chosen in the AFU's JSON file.
    logic clk;
    assign clk = `PLATFORM_PARAM_CCI_P_CLOCK;

    logic reset;
    assign reset = `PLATFORM_PARAM_CCI_P_RESET;


    // ====================================================================
    // Register signals at interface before consuming them
    // ====================================================================

    (* noprune *) logic [1:0]  cp2af_pwrState_T1;
    (* noprune *) logic        cp2af_error_T1;

    logic        reset_T1;
    t_if_ccip_Rx cp2af_sRx_T1;
    t_if_ccip_Tx af2cp_sTx_T0;

    ccip_interface_reg inst_green_ccip_interface_reg
       (
        .pClk                    (clk),
        .pck_cp2af_softReset_T0  (reset),
        .pck_cp2af_pwrState_T0   (pck_cp2af_pwrState),
        .pck_cp2af_error_T0      (pck_cp2af_error),
        .pck_cp2af_sRx_T0        (pck_cp2af_sRx),
        .pck_af2cp_sTx_T0        (af2cp_sTx_T0),

        .pck_cp2af_softReset_T1  (reset_T1),
        .pck_cp2af_pwrState_T1   (cp2af_pwrState_T1),
        .pck_cp2af_error_T1      (cp2af_error_T1),
        .pck_cp2af_sRx_T1        (cp2af_sRx_T1),
        .pck_af2cp_sTx_T1        (pck_af2cp_sTx)
        );


    // ====================================================================
    // User AFU goes here
    // ====================================================================

    //
    // vortex_afu depends on CCI-P and local memory being in the same
    // clock domain.  This is accomplished by choosing a common clock
    // in the AFU's JSON description.  The platform instantiates clock-
    // crossing shims automatically, as needed.
    //

    //
    // Memory banks are used very simply here.  Only bank is active at
    // a time, selected by mem_bank_select.  mem_bank_select is set
    // by a CSR from the host.
    //
    t_local_mem_byte_mask avs_byteenable;
    logic                 avs_waitrequest;
    t_local_mem_data      avs_readdata;
    logic                 avs_readdatavalid;
    t_local_mem_burst_cnt avs_burstcount;
    t_local_mem_data      avs_writedata;
    t_local_mem_addr      avs_address;
    logic                 avs_write;
    logic                 avs_read;

    // choose which memory bank to test
    logic [$clog2(NUM_LOCAL_MEM_BANKS)-1:0] mem_bank_select;

    vortex_afu #(
        .NUM_LOCAL_MEM_BANKS(NUM_LOCAL_MEM_BANKS)
    ) afu (
        .clk                 (clk),
        .reset               (reset_T1),

        .avs_writedata       (avs_writedata),
        .avs_readdata        (avs_readdata),
        .avs_address         (avs_address),
        .avs_waitrequest     (avs_waitrequest),
        .avs_write           (avs_write),
        .avs_read            (avs_read),
        .avs_byteenable      (avs_byteenable),
        .avs_burstcount      (avs_burstcount),
        .avs_readdatavalid   (avs_readdatavalid),
        .mem_bank_select     (mem_bank_select),

        .cp2af_sRxPort       (cp2af_sRx_T1),
        .af2cp_sTxPort       (af2cp_sTx_T0)
        );

    // 
    // Export the local memory interface signals as vectors so that bank
    // selection can use array syntax.
    //
    logic             avs_waitrequest_v[NUM_LOCAL_MEM_BANKS];
    t_local_mem_data  avs_readdata_v[NUM_LOCAL_MEM_BANKS];
    logic             avs_readdatavalid_v[NUM_LOCAL_MEM_BANKS];

    genvar b;
    generate
        for (b = 0; b < NUM_LOCAL_MEM_BANKS; b = b + 1)
        begin : lmb
            always_comb
            begin
                // Local memory to AFU signals
                avs_waitrequest_v[b] = local_mem[b].waitrequest;
                avs_readdata_v[b] = local_mem[b].readdata;
                avs_readdatavalid_v[b] = local_mem[b].readdatavalid;

                // Replicate address and write data to all banks.  Only
                // the request signals have to be bank-specific.
                local_mem[b].burstcount = avs_burstcount;
                local_mem[b].writedata = avs_writedata;
                local_mem[b].address = avs_address;
                local_mem[b].byteenable = avs_byteenable;

                // Request a write to this bank?
                local_mem[b].write = avs_write &&
                                     ($bits(mem_bank_select)'(b) == mem_bank_select);

                // Request a read from this bank?
                local_mem[b].read =  avs_read &&
                                     ($bits(mem_bank_select)'(b) == mem_bank_select);
            end
        end
    endgenerate

    assign avs_waitrequest   = avs_waitrequest_v[mem_bank_select];
    assign avs_readdata      = avs_readdata_v[mem_bank_select];
    assign avs_readdatavalid = avs_readdatavalid_v[mem_bank_select];

endmodule
