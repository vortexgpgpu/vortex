`include "VX_platform.vh"

`TRACING_OFF
module VX_matrix_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LOCK_ENABLE  = 0,
    parameter LOG_NUM_REQS = $clog2(NUM_REQS)
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire                     enable,
    input  wire [NUM_REQS-1:0]      requests,
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,   
    output wire                     grant_valid
  );

    if (NUM_REQS == 1)  begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        
        assign grant_index  = 0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin

        reg [NUM_REQS-1:1]  state [NUM_REQS-1:0];  
        wire [NUM_REQS-1:0] pri [NUM_REQS-1:0];
        wire [NUM_REQS-1:0] grant_unqual;
        
        for (genvar i = 0; i < NUM_REQS; i++) begin      
            for (genvar j = 0; j < NUM_REQS; j++) begin
                if (j > i) begin
                    assign pri[j][i] = requests[i] && state[i][j];
                end 
                else if (j < i) begin
                    assign pri[j][i] = requests[i] && !state[j][i];
                end 
                else begin
                    assign pri[j][i] = 0;            
                end
            end
            assign grant_unqual[i] = requests[i] && !(| pri[i]);
        end
        
        for (genvar i = 0; i < NUM_REQS; i++) begin      
            for (genvar j = i + 1; j < NUM_REQS; j++) begin
                always @(posedge clk) begin                       
                    if (reset) begin         
                        state[i][j] <= 0;
                    end else begin
                        state[i][j] <= (state[i][j] || grant_unqual[j]) && !grant_unqual[i];
                    end
                end
            end
        end

        if (LOCK_ENABLE == 0) begin
            `UNUSED_VAR (enable)
            assign grant_onehot = grant_unqual;
        end else begin
            reg [NUM_REQS-1:0] grant_unqual_prev;
            always @(posedge clk) begin
                if (reset) begin
                    grant_unqual_prev <= 0;
                end else if (enable) begin
                    grant_unqual_prev <= grant_unqual;
                end
            end
            assign grant_onehot = enable ? grant_unqual : grant_unqual_prev;
        end

        VX_onehot_encoder #(
            .N (NUM_REQS)
        ) encoder (
            .data_in    (grant_unqual),
            .data_out   (grant_index),
            `UNUSED_PIN (valid_out)
        );

        assign grant_valid = (| requests);

    end
    
endmodule
`TRACING_ON
