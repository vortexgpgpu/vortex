`include "VX_platform.vh"

`TRACING_OFF
module VX_popcount #(
    parameter MODEL = 1,
    parameter N     = 1,
    parameter M     = $clog2(N+1)  
) (
    input  wire [N-1:0] data_in,
    output wire [M-1:0] data_out
);
    `UNUSED_PARAM (MODEL)    

`ifndef SYNTHESIS
    assign data_out = $countones(data_in);
`else
`ifdef QUARTUS
    assign data_out = $countones(data_in);
`else
    if (N == 1) begin

        assign data_out = data_in;

    end else if (MODEL == 1) begin
    `IGNORE_WARNINGS_BEGIN
        localparam PN    = 1 << $clog2(N);
        localparam LOGPN = $clog2(PN);
        
        wire [M-1:0] tmp [0:PN-1] [0:PN-1];
        
        for (genvar i = 0; i < N; ++i) begin        
            assign tmp[0][i] = data_in[i];
        end

        for (genvar i = N; i < PN; ++i) begin        
            assign tmp[0][i] = '0;
        end

        for (genvar j = 0; j < LOGPN; ++j) begin
            for (genvar i = 0; i < (1 << (LOGPN-j-1)); ++i) begin
                assign tmp[j+1][i] = tmp[j][i*2] + tmp[j][i*2+1];
            end
        end

        assign data_out = tmp[LOGPN][0];
    `IGNORE_WARNINGS_END
    end else begin

        reg [M-1:0] cnt_r;

        always @(*) begin
            cnt_r = '0;
            for (integer i = 0; i < N; ++i) begin
            `IGNORE_WARNINGS_BEGIN
                cnt_r = cnt_r + data_in[i];
            `IGNORE_WARNINGS_END
            end
        end

        assign data_out = cnt_r;
    
    end
`endif
`endif

endmodule
`TRACING_ON
