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
`elsif QUARTUS
    assign data_out = $countones(data_in);
`else
    if (N == 1) begin

        assign data_out = data_in;

    end else if (MODEL == 1) begin

        localparam PN = 1 << $clog2(N);
        localparam LOGPN = $clog2(PN);

    `IGNORE_UNOPTFLAT_BEGIN
        wire [M-1:0] tmp [LOGPN-1:0][PN-1:0];
    `IGNORE_UNOPTFLAT_END

        for (genvar j = 0; j < LOGPN; ++j) begin
            localparam D = j + 1;
            localparam Q = (D < LOGPN) ? (D + 1) : M;
            for (genvar i = 0; i < (1 << (LOGPN-j-1)); ++i) begin                
                localparam l = i * 2;
                localparam r = i * 2 + 1;
                wire [Q-1:0] res;
                if (j == 0) begin
                    if (r < N) begin
                        assign res = data_in[l] + data_in[r];
                    end else if (l < N) begin
                        assign res = 2'(data_in[l]);
                    end else begin
                        assign res = 2'b0;
                    end
                end else begin
                    assign res = D'(tmp[j-1][l]) + D'(tmp[j-1][r]);
                end
                assign tmp[j][i] = M'(res);
            end
        end

        assign data_out = tmp[LOGPN-1][0];
    
    end else begin

        reg [M-1:0] cnt_r;

        always @(*) begin
            cnt_r = '0;
            for (integer i = 0; i < N; ++i) begin
                cnt_r = cnt_r + M'(data_in[i]);
            end
        end

        assign data_out = cnt_r;
    
    end
`endif

endmodule
`TRACING_ON
