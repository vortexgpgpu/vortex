`include "VX_platform.vh"

module VX_stream_arbiter #(
    parameter NUM_REQS    = 1,
    parameter DATAW       = 1,
    parameter TYPE        = "R",
    parameter LOCK_ENABLE = 1,
    parameter BUFFERED    = 0
) (
    input  wire clk,
    input  wire reset,

    input  wire [NUM_REQS-1:0]            valid_in,
    input  wire [NUM_REQS-1:0][DATAW-1:0] data_in,
    output wire [NUM_REQS-1:0]            ready_in,

    output wire             valid_out,
    output wire [DATAW-1:0] data_out,    
    input  wire             ready_out
);
  
    localparam LOG_NUM_REQS = $clog2(NUM_REQS);

    if (NUM_REQS > 1)  begin
        wire                    sel_valid;
        wire                    sel_ready;
        wire [NUM_REQS-1:0]     sel_1hot;

        if (TYPE == "X") begin
            VX_fixed_arbiter #(
                .NUM_REQS(NUM_REQS),
                .LOCK_ENABLE(LOCK_ENABLE)
            ) sel_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in),  
                .enable       (sel_ready),     
                .grant_valid  (sel_valid),
                .grant_onehot (sel_1hot),
				`UNUSED_PIN (grant_index)
            );
        end else if (TYPE == "R") begin
            VX_rr_arbiter #(
                .NUM_REQS(NUM_REQS),
                .LOCK_ENABLE(LOCK_ENABLE)
            ) sel_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in),  
                .enable       (sel_ready),
                .grant_valid  (sel_valid),
                .grant_onehot (sel_1hot),
				`UNUSED_PIN (grant_index)
            );
        end else if (TYPE == "F") begin
            VX_fair_arbiter #(
                .NUM_REQS(NUM_REQS),
                .LOCK_ENABLE(LOCK_ENABLE)
            ) sel_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in),  
                .enable       (sel_ready),     
                .grant_valid  (sel_valid),
                .grant_onehot (sel_1hot),
				`UNUSED_PIN (grant_index)
            );
        end else if (TYPE == "M") begin
            VX_matrix_arbiter #(
                .NUM_REQS(NUM_REQS),
                .LOCK_ENABLE(LOCK_ENABLE)
            ) sel_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in),  
                .enable       (sel_ready),     
                .grant_valid  (sel_valid),
                .grant_onehot (sel_1hot),
				`UNUSED_PIN (grant_index)
            );
        end else begin
            $error ("invalid parameter");
        end

        wire [DATAW-1:0] data_in_sel;

        VX_onehot_mux #(
            .DATAW (DATAW),
            .N     (NUM_REQS)
        ) data_in_mux (
            .data_in  (data_in),
            .sel_in   (sel_1hot),
            .data_out (data_in_sel)
        );

        VX_skid_buffer #(
            .DATAW    (DATAW),
            .PASSTHRU (!BUFFERED)
        ) out_buffer (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (sel_valid),        
            .data_in   (data_in_sel),
            .ready_in  (sel_ready),      
            .valid_out (valid_out),
            .data_out  (data_out),
            .ready_out (ready_out)
        );

        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign ready_in[i] = sel_1hot[i] && sel_ready;
        end

    end else begin
    
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        
        assign valid_out = valid_in;        
        assign data_out  = data_in;
        assign ready_in  = ready_out;

    end
    
endmodule