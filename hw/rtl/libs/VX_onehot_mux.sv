`include "VX_platform.vh"

`TRACING_OFF
module VX_onehot_mux #(
    parameter DATAW = 1,
    parameter N     = 1,
    parameter MODEL = 1
) (
    input wire [N-1:0][DATAW-1:0] data_in,    
    input wire [N-1:0]            sel_in,    
    output wire [DATAW-1:0]       data_out
); 
    if (N > 1) begin
        if (MODEL == 1) begin
            for (genvar i = 0; i < N; ++i) begin
                assign data_out = sel_in[i] ? data_in[i] : 'z;
            end
        end else if (MODEL == 2) begin           
            reg [DATAW-1:0] data_out_r;
            always @(*) begin
                data_out_r = '0;
                for (integer i = 0; i < N; ++i) begin
                    data_out_r |= {DATAW{sel_in[i]}} & data_in[i];
                end
            end
            assign data_out = data_out_r; 
        end else if (MODEL == 3) begin           
            wire [N-1:0][DATAW-1:0] mask;
            for (genvar i = 0; i < N; ++i) begin
                assign mask[i] = {DATAW{sel_in[i]}} & data_in[i];
            end            
            for (genvar i = 0; i < DATAW; ++i) begin
                wire [N-1:0] gather;
                for (genvar j = 0; j < N; ++j) begin
                    assign gather[j] = mask[j][i];
                end
                assign data_out[i] = (| gather);
            end       
        end else begin
            reg [DATAW-1:0] data_out_r;
            always @(*) begin
                data_out_r = 'x;
                for (integer i = N-1; i >= 0; --i) begin
                    if (sel_in[i]) begin
                        data_out_r = data_in[i];
                    end
                end
            end
            assign data_out = data_out_r; 
        end
    end else begin
        `UNUSED_VAR (sel_in)
        assign data_out = data_in;
    end

endmodule
`TRACING_ON
