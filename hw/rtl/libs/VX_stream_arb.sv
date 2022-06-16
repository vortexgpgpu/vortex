`include "VX_platform.vh"

`TRACING_OFF
module VX_stream_arb #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter NUM_LANES      = 1,
    parameter DATAW          = 1,
    parameter string ARBITER = "P",
    parameter LOCK_ENABLE    = 1,
    parameter MAX_FANOUT     = 8,
    parameter BUFFERED       = 0    
) (
    input  wire clk,
    input  wire reset,

    input  wire [NUM_INPUTS-1:0][NUM_LANES-1:0]             valid_in,
    input  wire [NUM_INPUTS-1:0][NUM_LANES-1:0][DATAW-1:0]  data_in,
    output wire [NUM_INPUTS-1:0][NUM_LANES-1:0]             ready_in,

    output wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            valid_out,
    output wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0][DATAW-1:0] data_out,    
    input  wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            ready_out
);
    localparam NUM_REQS     = (NUM_INPUTS > NUM_OUTPUTS) ? ((NUM_INPUTS + NUM_OUTPUTS - 1) / NUM_OUTPUTS) : ((NUM_OUTPUTS + NUM_INPUTS - 1) / NUM_INPUTS);
    localparam LOG_NUM_REQS = `CLOG2(NUM_REQS);
    localparam NUM_REQS_A   = 1 << LOG_NUM_REQS;

    if (NUM_INPUTS > NUM_OUTPUTS) begin

        if (MAX_FANOUT !=0 && (NUM_INPUTS > MAX_FANOUT) && NUM_OUTPUTS > 1) begin
            
            for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin

                localparam BATCH_BEGIN = i * NUM_REQS;
                localparam BATCH_END   = `MIN(BATCH_BEGIN + NUM_REQS, NUM_INPUTS);
                localparam BATCH_SIZE  = BATCH_END - BATCH_BEGIN;

                VX_stream_arb #(
                    .NUM_INPUTS  (BATCH_SIZE),
                    .NUM_OUTPUTS (1),   
                    .NUM_LANES   (NUM_LANES),     
                    .DATAW       (DATAW),
                    .ARBITER     (ARBITER),
                    .LOCK_ENABLE (LOCK_ENABLE),
                    .MAX_FANOUT  (MAX_FANOUT),
                    .BUFFERED    (BUFFERED)
                ) arb_slice (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_in[BATCH_END-1: BATCH_BEGIN]),
                    .data_in   (data_in[BATCH_END-1: BATCH_BEGIN]),
                    .ready_in  (ready_in[BATCH_END-1: BATCH_BEGIN]),   
                    .valid_out (valid_out[i]),
                    .data_out  (data_out[i]),
                    .ready_out (ready_out[i])
                );
            end

        end else if (MAX_FANOUT !=0 && (NUM_INPUTS > MAX_FANOUT) && NUM_OUTPUTS == 1) begin

            localparam BATCHES = (NUM_INPUTS + MAX_FANOUT - 1) / MAX_FANOUT;

            wire [BATCHES-1:0][NUM_LANES-1:0]            valid_tmp;
            wire [BATCHES-1:0][NUM_LANES-1:0][DATAW-1:0] data_tmp;
            wire [BATCHES-1:0][NUM_LANES-1:0]            ready_tmp;
            
            for (genvar i = 0; i < BATCHES; ++i) begin

                localparam BATCH_BEGIN = i * MAX_FANOUT;
                localparam BATCH_END   = `MIN(BATCH_BEGIN + MAX_FANOUT, NUM_INPUTS);
                localparam BATCH_SIZE  = BATCH_END - BATCH_BEGIN;

                VX_stream_arb #(
                    .NUM_INPUTS  (BATCH_SIZE),
                    .NUM_OUTPUTS (1),   
                    .NUM_LANES   (NUM_LANES),     
                    .DATAW       (DATAW),
                    .ARBITER     (ARBITER),
                    .LOCK_ENABLE (LOCK_ENABLE),
                    .MAX_FANOUT  (MAX_FANOUT),
                    .BUFFERED    (2)
                ) arb_slice (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_in[BATCH_END-1: BATCH_BEGIN]),
                    .data_in   (data_in[BATCH_END-1: BATCH_BEGIN]),
                    .ready_in  (ready_in[BATCH_END-1: BATCH_BEGIN]),   
                    .valid_out (valid_tmp[i]),   
                    .data_out  (data_tmp[i]),
                    .ready_out (ready_tmp[i])
                );
            end

            VX_stream_arb #(
                .NUM_INPUTS  (BATCHES),
                .NUM_OUTPUTS (1),   
                .NUM_LANES   (NUM_LANES),     
                .DATAW       (DATAW),
                .ARBITER     (ARBITER),
                .LOCK_ENABLE (LOCK_ENABLE),
                .MAX_FANOUT  (MAX_FANOUT),
                .BUFFERED    (BUFFERED)
            ) arb_top (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_tmp),
                .data_in   (data_tmp),
                .ready_in  (ready_tmp),   
                .valid_out (valid_out),   
                .data_out  (data_out),
                .ready_out (ready_out)
            );

        end else begin

            wire [NUM_REQS_A-1:0][NUM_OUTPUTS-1:0][NUM_LANES-1:0]             valid_in_r;
            wire [NUM_REQS_A-1:0][NUM_OUTPUTS-1:0][NUM_LANES-1:0][DATAW-1:0]  data_in_r;

            for (genvar i = 0; i < NUM_REQS_A; ++i) begin
                for (genvar j = 0; j < NUM_OUTPUTS; ++j) begin
                    localparam ii = i * NUM_OUTPUTS + j;
                    if (ii < NUM_INPUTS) begin
                        assign valid_in_r[i][j] = valid_in[ii];
                        assign data_in_r[i][j]  = data_in[ii];
                    end else begin
                        assign valid_in_r[i][j] = 0;
                        assign data_in_r[i][j]  = 'x;
                        wire [NUM_LANES-1:0] tmp = valid_in_r[i][j];
                        `UNUSED_VAR (tmp)
                    end
                end
            end        

            wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            valid_out_r;
            wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0][DATAW-1:0] data_out_r;
            wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            ready_out_r;
        
            wire [NUM_REQS-1:0]     arb_requests;
            wire                    arb_valid;
            wire [LOG_NUM_REQS-1:0] arb_index;
            wire [NUM_REQS-1:0]     arb_onehot;
            wire                    arb_unlock;

            for (genvar i = 0; i < NUM_REQS; ++i) begin
                assign arb_requests[i] = (| valid_in_r[i]);
            end

            VX_generic_arbiter #(
                .NUM_REQS    (NUM_REQS),
                .LOCK_ENABLE (1),
                .TYPE        (ARBITER)
            ) arbiter (
                .clk          (clk),
                .reset        (reset),
                .requests     (arb_requests),  
                .unlock       (arb_unlock),
                .grant_valid  (arb_valid),
                .grant_index  (arb_index),
                .grant_onehot (arb_onehot)
            );

            if ((NUM_OUTPUTS * NUM_LANES) > 1) begin
                `UNUSED_VAR (arb_valid)
                assign valid_out_r = valid_in_r[arb_index];
                assign arb_unlock  = | (valid_out_r & ready_out_r);
            end else begin            
                assign valid_out_r = arb_valid;
                assign arb_unlock  = valid_out_r & ready_out_r;
            end

            assign data_out_r = data_in_r[arb_index];

            for (genvar i = 0; i < NUM_REQS; ++i) begin
                for (genvar j = 0; j < NUM_OUTPUTS; ++j) begin
                    localparam ii = i * NUM_OUTPUTS + j;                
                    if (ii < NUM_INPUTS) begin                    
                        assign ready_in[ii] = ready_out_r[j] & {NUM_LANES{arb_onehot[i]}};
                    end
                end
            end

            for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
                for (genvar j = 0; j < NUM_LANES; ++j) begin
                    VX_skid_buffer #(
                        .DATAW    (DATAW),
                        .PASSTHRU (BUFFERED == 0),
                        .OUT_REG  (BUFFERED > 1)
                    ) out_buffer (
                        .clk       (clk),
                        .reset     (reset),
                        .valid_in  (valid_out_r[i][j]),
                        .data_in   (data_out_r[i][j]),
                        .ready_in  (ready_out_r[i][j]),
                        .valid_out (valid_out[i][j]),
                        .data_out  (data_out[i][j]),
                        .ready_out (ready_out[i][j])
                    );
                end
            end
        end

    end else if (NUM_OUTPUTS > NUM_INPUTS) begin

        if (MAX_FANOUT !=0 && (NUM_OUTPUTS > MAX_FANOUT) && NUM_INPUTS > 1) begin

            for (genvar i = 0; i < NUM_INPUTS; ++i) begin

                localparam BATCH_BEGIN = i * NUM_REQS;
                localparam BATCH_END   = `MIN(BATCH_BEGIN + NUM_REQS, NUM_OUTPUTS);
                localparam BATCH_SIZE  = BATCH_END - BATCH_BEGIN;

                VX_stream_arb #(
                    .NUM_INPUTS  (1),
                    .NUM_OUTPUTS (BATCH_SIZE),
                    .NUM_LANES   (NUM_LANES),     
                    .DATAW       (DATAW),
                    .ARBITER     (ARBITER),
                    .LOCK_ENABLE (LOCK_ENABLE),
                    .MAX_FANOUT  (MAX_FANOUT),
                    .BUFFERED    (BUFFERED)
                ) arb_slice (
                    .clk       (clk),
                    .reset     (reset),                    
                    .valid_in  (valid_in[i]),
                    .data_in   (data_in[i]),
                    .ready_in  (ready_in[i]),
                    .valid_out (valid_out[BATCH_END-1: BATCH_BEGIN]),
                    .data_out  (data_out[BATCH_END-1: BATCH_BEGIN]),
                    .ready_out (ready_out[BATCH_END-1: BATCH_BEGIN])
                );
            end

        end else if (MAX_FANOUT !=0 && (NUM_OUTPUTS > MAX_FANOUT) && NUM_INPUTS == 1) begin

            localparam BATCHES = (NUM_OUTPUTS + MAX_FANOUT - 1) / MAX_FANOUT;

            wire [BATCHES-1:0][NUM_LANES-1:0]            valid_tmp;
            wire [BATCHES-1:0][NUM_LANES-1:0][DATAW-1:0] data_tmp;
            wire [BATCHES-1:0][NUM_LANES-1:0]            ready_tmp;

            VX_stream_arb #(
                .NUM_INPUTS  (1),
                .NUM_OUTPUTS (BATCHES),
                .NUM_LANES   (NUM_LANES),     
                .DATAW       (DATAW),
                .ARBITER     (ARBITER),
                .LOCK_ENABLE (LOCK_ENABLE),
                .MAX_FANOUT  (MAX_FANOUT),
                .BUFFERED    (BUFFERED)
            ) arb_top (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_in),
                .data_in   (data_in),
                .ready_in  (ready_in),   
                .valid_out (valid_tmp),   
                .data_out  (data_tmp),
                .ready_out (ready_tmp)
            );
            
            for (genvar i = 0; i < BATCHES; ++i) begin

                localparam BATCH_BEGIN = i * MAX_FANOUT;
                localparam BATCH_END   = `MIN(BATCH_BEGIN + MAX_FANOUT, NUM_INPUTS);
                localparam BATCH_SIZE  = BATCH_END - BATCH_BEGIN;

                VX_stream_arb #(
                    .NUM_INPUTS  (1),
                    .NUM_OUTPUTS (BATCH_SIZE),   
                    .NUM_LANES   (NUM_LANES),     
                    .DATAW       (DATAW),
                    .ARBITER     (ARBITER),
                    .LOCK_ENABLE (LOCK_ENABLE),
                    .MAX_FANOUT  (MAX_FANOUT),
                    .BUFFERED    (2)
                ) arb_slice (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_tmp[i]),
                    .data_in   (data_tmp[i]),
                    .ready_in  (ready_tmp[i]),
                    .valid_out (valid_out[BATCH_END-1: BATCH_BEGIN]),
                    .data_out  (data_out[BATCH_END-1: BATCH_BEGIN]),
                    .ready_out (ready_out[BATCH_END-1: BATCH_BEGIN])
                );
            end

        end else begin
    
            wire [NUM_REQS_A-1:0][NUM_INPUTS-1:0][NUM_LANES-1:0] ready_out_r;        
        
            wire [NUM_REQS-1:0]     arb_requests;
            wire                    arb_valid;
            wire [LOG_NUM_REQS-1:0] arb_index;
            wire [NUM_REQS-1:0]     arb_onehot;
            wire                    arb_unlock;

            for (genvar i = 0; i < NUM_REQS; ++i) begin
                assign arb_requests[i] = (| ready_out_r[i]);
            end

            VX_generic_arbiter #(
                .NUM_REQS    (NUM_REQS),
                .LOCK_ENABLE (LOCK_ENABLE),
                .TYPE        (ARBITER)
            ) arbiter (
                .clk          (clk),
                .reset        (reset),
                .requests     (arb_requests),  
                .unlock       (arb_unlock),
                .grant_valid  (arb_valid),
                .grant_index  (arb_index),
                .grant_onehot (arb_onehot)
            );

            if ((NUM_INPUTS * NUM_LANES) > 1) begin
                `UNUSED_VAR (arb_valid)
                assign ready_in   = ready_out_r[arb_index];
                assign arb_unlock = | (valid_in & ready_in);
            end else begin
                `UNUSED_VAR (arb_index)
                assign ready_in   = arb_valid;
                assign arb_unlock = valid_in & ready_in;
            end

            for (genvar i = 0; i < NUM_REQS_A; ++i) begin
                for (genvar j = 0; j < NUM_INPUTS; ++j) begin
                    localparam ii = i * NUM_INPUTS + j;
                    if (ii < NUM_OUTPUTS) begin
                        for (genvar k = 0; k < NUM_LANES; ++k) begin                    
                            VX_skid_buffer #(
                                .DATAW    (DATAW),
                                .PASSTHRU (BUFFERED == 0),
                                .OUT_REG  (BUFFERED > 1)
                            ) out_buffer (
                                .clk       (clk),
                                .reset     (reset),
                                .valid_in  (valid_in[j][k] && arb_onehot[i]),
                                .data_in   (data_in[j][k]),
                                .ready_in  (ready_out_r[i][j][k]),
                                .valid_out (valid_out[ii][k]),
                                .data_out  (data_out[ii][k]),
                                .ready_out (ready_out[ii][k])
                            );
                        end
                    end else begin
                        assign ready_out_r[i][j] = '0;                                        
                        wire [NUM_LANES-1:0] tmp = ready_out_r[i][j];
                        `UNUSED_VAR (tmp)
                    end                
                end
            end
        end
    
    end else begin

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            for (genvar j = 0; j < NUM_LANES; ++j) begin                
                VX_skid_buffer #(
                    .DATAW    (DATAW),
                    .PASSTHRU (BUFFERED == 0),
                    .OUT_REG  (BUFFERED > 1)
                ) out_buffer (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_in[i][j]),
                    .data_in   (data_in[i][j]),
                    .ready_in  (ready_in[i][j]),      
                    .valid_out (valid_out[i][j]),
                    .data_out  (data_out[i][j]),
                    .ready_out (ready_out[i][j])
                );
            end
        end
    end
    
endmodule
`TRACING_ON
