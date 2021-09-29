`include "VX_platform.vh"

module VX_stream_arbiter #(
    parameter NUM_REQS    = 1,
    parameter LANES       = 1,
    parameter DATAW       = 1,
    parameter TYPE        = "P",
    parameter LOCK_ENABLE = 1,
    parameter BUFFERED    = 0
) (
    input  wire clk,
    input  wire reset,

    input  wire [NUM_REQS-1:0][LANES-1:0]            valid_in,
    input  wire [NUM_REQS-1:0][LANES-1:0][DATAW-1:0] data_in,
    output wire [NUM_REQS-1:0][LANES-1:0]            ready_in,

    output wire [LANES-1:0]            valid_out,
    output wire [LANES-1:0][DATAW-1:0] data_out,    
    input  wire [LANES-1:0]            ready_out
);
    localparam LOG_NUM_REQS = `CLOG2(NUM_REQS);

    if (NUM_REQS > 1)  begin
        wire                    sel_valid;
        wire                    sel_ready;
        wire [LOG_NUM_REQS-1:0] sel_index;
        wire [NUM_REQS-1:0]     sel_onehot;

        wire [NUM_REQS-1:0] valid_in_any;
        wire [LANES-1:0] ready_in_sel;

        if (LANES > 1) begin
            for (genvar i = 0; i < NUM_REQS; i++) begin
                assign valid_in_any[i] = (| valid_in[i]);
            end
            assign sel_ready = (| ready_in_sel);
        end else begin
            for (genvar i = 0; i < NUM_REQS; i++) begin
                assign valid_in_any[i] = valid_in[i];
            end
            assign sel_ready = ready_in_sel;
        end

        if (TYPE == "P") begin
            VX_fixed_arbiter #(
                .NUM_REQS    (NUM_REQS),
                .LOCK_ENABLE (LOCK_ENABLE)
            ) sel_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in_any),  
                .enable       (sel_ready),
                .grant_valid  (sel_valid),
				.grant_index  (sel_index),
                .grant_onehot (sel_onehot)
            );
        end else if (TYPE == "R") begin
            VX_rr_arbiter #(
                .NUM_REQS    (NUM_REQS),
                .LOCK_ENABLE (LOCK_ENABLE)
            ) sel_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in_any),  
                .enable       (sel_ready),
                .grant_valid  (sel_valid),
				.grant_index  (sel_index),
                .grant_onehot (sel_onehot)
            );
        end else if (TYPE == "F") begin
            VX_fair_arbiter #(
                .NUM_REQS    (NUM_REQS),
                .LOCK_ENABLE (LOCK_ENABLE)
            ) sel_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in_any),  
                .enable       (sel_ready),     
                .grant_valid  (sel_valid),
				.grant_index  (sel_index),
                .grant_onehot (sel_onehot)
            );
        end else if (TYPE == "M") begin
            VX_matrix_arbiter #(
                .NUM_REQS    (NUM_REQS),
                .LOCK_ENABLE (LOCK_ENABLE)
            ) sel_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in_any),  
                .enable       (sel_ready),     
                .grant_valid  (sel_valid),
				.grant_index  (sel_index),
                .grant_onehot (sel_onehot)
            );
        end else begin
            `ERROR(("invalid parameter"));
        end

        wire [LANES-1:0] valid_in_sel;
        wire [LANES-1:0][DATAW-1:0] data_in_sel;

        if (LANES > 1) begin
            wire [NUM_REQS-1:0][(LANES * (1 + DATAW))-1:0] valid_data_in;
            for (genvar i = 0; i < NUM_REQS; i++) begin
                assign valid_data_in[i] = {valid_in[i], data_in[i]};
            end
            assign {valid_in_sel, data_in_sel} = valid_data_in[sel_index];
            `UNUSED_VAR (sel_valid)
        end else begin
            assign data_in_sel  = data_in[sel_index];            
            assign valid_in_sel = sel_valid;
        end

        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign ready_in[i] = ready_in_sel & {LANES{sel_onehot[i]}};
        end

        for (genvar i = 0; i < LANES; ++i) begin
            VX_skid_buffer #(
                .DATAW    (DATAW),
                .PASSTHRU (0 == BUFFERED),
                .OUT_REG  (2 == BUFFERED)
            ) out_buffer (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_in_sel[i]),        
                .data_in   (data_in_sel[i]),
                .ready_in  (ready_in_sel[i]),      
                .valid_out (valid_out[i]),
                .data_out  (data_out[i]),
                .ready_out (ready_out[i])
            );
        end

    end else begin
    
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        
        assign valid_out = valid_in;        
        assign data_out  = data_in;
        assign ready_in  = ready_out;

    end
    
endmodule