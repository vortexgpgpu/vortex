`include "VX_platform.vh"

`TRACING_OFF
module VX_find_first #(
    parameter N       = 1,
    parameter DATAW   = 1,
    parameter REVERSE = 0,
    localparam LOGN   = $clog2(N)
) (
    input  wire [N-1:0][DATAW-1:0] data_i,
    input  wire [N-1:0]            valid_i,    
    output wire [DATAW-1:0]        data_o,
    output wire                    valid_o
);
    if (N > 1) begin    
        wire [N-1:0] valid_r;
        wire [N-1:0][DATAW-1:0] data_r;

        for (genvar i = 0; i < N; ++i) begin
            assign valid_r[i] = REVERSE ? valid_i[N-1-i] : valid_i[i];
            assign data_r[i]  = REVERSE ? data_i[N-1-i] : data_i[i];
        end

    `IGNORE_WARNINGS_BEGIN
        wire [2**LOGN-1:0]            s_n;
        wire [2**LOGN-1:0][DATAW-1:0] d_n;       
    `IGNORE_WARNINGS_END

        for (genvar i = 0; i < LOGN; ++i) begin
            if (i == (LOGN-1)) begin
                for (genvar j = 0; j < 2**i; ++j) begin
                    if ((j*2) < (N-1)) begin
                        assign s_n[2**i-1+j] = valid_r[j*2] | valid_r[j*2+1];
                        assign d_n[2**i-1+j] = valid_r[j*2] ? data_r[j*2] : data_r[j*2+1];
                    end
                    if ((j*2) == (N-1)) begin
                        assign s_n[2**i-1+j] = valid_r[j*2];
                        assign d_n[2**i-1+j] = data_r[j*2];
                    end
                    if ((j*2) > (N-1)) begin
                        assign s_n[2**i-1+j] = 0;
                        assign d_n[2**i-1+j] = 'x;
                    end
                end
            end else begin
                for (genvar j = 0; j < 2**i; ++j) begin
                    assign s_n[2**i-1+j] = s_n[2**(i+1)-1+j*2] | s_n[2**(i+1)-1+j*2+1];
                    assign d_n[2**i-1+j] = s_n[2**(i+1)-1+j*2] ? d_n[2**(i+1)-1+j*2] : d_n[2**(i+1)-1+j*2+1];
                end
            end
        end     
        
        assign valid_o = s_n[0];
        assign data_o  = d_n[0];  
    end else begin
        assign valid_o = valid_i;
        assign data_o  = data_i[0];  
    end    
  
endmodule
`TRACING_ON