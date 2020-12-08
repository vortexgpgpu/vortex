`include "VX_platform.vh"

module VX_stream_arbiter #(
    parameter NUM_REQS   = 1,
    parameter DATAW      = 1,
    parameter TYPE       = "R",
    parameter IN_BUFFER  = 0,
    parameter OUT_BUFFER = 0
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

        wire [NUM_REQS-1:0]            valid_in_qual;
        wire [NUM_REQS-1:0][DATAW-1:0] data_in_qual;
        wire [NUM_REQS-1:0]            ready_in_qual;

        for (genvar i = 0; i < NUM_REQS; ++i) begin
            VX_skid_buffer #(
                .DATAW    (DATAW),
                .PASSTHRU (!IN_BUFFER)
            ) req_buffer (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_in[i]),        
                .data_in   (data_in[i]),
                .ready_in  (ready_in[i]),        
                .valid_out (valid_in_qual[i]),
                .data_out  (data_in_qual[i]),
                .ready_out (ready_in_qual[i])
            );
        end

        wire                    sel_enable;
        wire                    sel_valid;
        wire [LOG_NUM_REQS-1:0] sel_idx;
        wire [NUM_REQS-1:0]     sel_1hot;

        if (TYPE == "X") begin

            VX_fixed_arbiter #(
                .NUM_REQS(NUM_REQS),
                .LOCK_ENABLE(1)
            ) sel_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in_qual),  
                .enable       (sel_enable),      
                .grant_valid  (sel_valid),
                .grant_index  (sel_idx),
                .grant_onehot (sel_1hot)
            );

        end else if (TYPE == "R") begin

            VX_rr_arbiter #(
                .NUM_REQS(NUM_REQS),
                .LOCK_ENABLE(1)
            ) sel_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in_qual),  
                .enable       (sel_enable),      
                .grant_valid  (sel_valid),
                .grant_index  (sel_idx),
                .grant_onehot (sel_1hot)
            );

        end else if (TYPE == "F") begin

            VX_fair_arbiter #(
                .NUM_REQS(NUM_REQS),
                .LOCK_ENABLE(1)
            ) sel_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in_qual),  
                .enable       (sel_enable),      
                .grant_valid  (sel_valid),
                .grant_index  (sel_idx),
                .grant_onehot (sel_1hot)
            );

        end else if (TYPE == "M") begin

            VX_matrix_arbiter #(
                .NUM_REQS(NUM_REQS),
                .LOCK_ENABLE(1)
            ) sel_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (valid_in_qual),  
                .enable       (sel_enable),      
                .grant_valid  (sel_valid),
                .grant_index  (sel_idx),
                .grant_onehot (sel_1hot)
            );

        end

        if (OUT_BUFFER) begin

            wire stall = ~ready_out && valid_out;
            assign sel_enable = ~stall;

            VX_generic_register #(
                .N(1 + DATAW),
                .R(1)
            ) pipe_reg (
                .clk      (clk),
                .reset    (reset),
                .stall    (stall),
                .flush    (1'b0),
                .data_in  ({sel_valid, data_in_qual[sel_idx]}),
                .data_out ({valid_out, data_out})
            );

            for (genvar i = 0; i < NUM_REQS; i++) begin
                assign ready_in_qual[i] = sel_1hot[i] && ~stall;                
            end
            
        end else begin

            assign sel_enable = ready_out;
            assign valid_out  = sel_valid;   
            assign data_out   = data_in_qual[sel_idx];

            for (genvar i = 0; i < NUM_REQS; i++) begin
                assign ready_in_qual[i] = sel_1hot[i] && ready_out;
            end

        end       

    end else begin
    
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        
        assign valid_out = valid_in;        
        assign data_out  = data_in;
        assign ready_in  = ready_out;        

    end
    
endmodule