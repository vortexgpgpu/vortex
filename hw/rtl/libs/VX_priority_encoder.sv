`include "VX_platform.vh"

`TRACING_OFF
module VX_priority_encoder #( 
    parameter N       = 1,  
    parameter REVERSE = 0,
    parameter MODEL   = 1,
    parameter LN      = `LOG2UP(N)
) (
    input  wire [N-1:0]  data_in,  
    output wire [N-1:0]  onehot,
    output wire [LN-1:0] index,
    output wire          valid_out
);
    wire [N-1:0] reversed; 

    if (REVERSE) begin
        for (genvar i = 0; i < N; ++i) begin
            assign reversed[N-i-1] = data_in[i];
        end        
    end else begin
        assign reversed = data_in;
    end

    if (N == 1) begin

        assign onehot    = reversed;
        assign index     = 0;
        assign valid_out = reversed;

    end else if (N == 2) begin

        assign onehot    = {~reversed[0], reversed[0]};
        assign index     = ~reversed[0];
        assign valid_out = (| reversed);

    end else if (MODEL == 1) begin

        wire [N-1:0] scan_lo;

        VX_scan #(
            .N  (N),
            .OP (2)
        ) scan (
            .data_in  (reversed),
            .data_out (scan_lo)
        );

        VX_lzc #(
            .N (N)
        ) lzc (
            .in_i  (reversed),            
            .cnt_o (index),
            `UNUSED_PIN (valid_o)
        );

        assign onehot    = scan_lo & {(~scan_lo[N-2:0]), 1'b1};
        assign valid_out = scan_lo[N-1];

    end else if (MODEL == 2) begin

    `IGNORE_WARNINGS_BEGIN
        wire [N-1:0] higher_pri_regs;
    `IGNORE_WARNINGS_END
        assign higher_pri_regs[N-1:1] = higher_pri_regs[N-2:0] | reversed[N-2:0];
        assign higher_pri_regs[0]     = 1'b0;
        assign onehot[N-1:0] = reversed[N-1:0] & ~higher_pri_regs[N-1:0];

        VX_lzc #(
            .N (N)
        ) lzc (
            .in_i    (reversed),            
            .cnt_o   (index),
            .valid_o (valid_out)
        );

    end else if (MODEL == 3) begin

        assign onehot = reversed & ~(reversed-1);

        VX_lzc #(
            .N (N)
        ) lzc (
            .in_i    (reversed),           
            .cnt_o   (index),
            .valid_o (valid_out)
        );

    end else begin

        reg [LN-1:0] index_r;
        reg [N-1:0]  onehot_r;

        always @(*) begin
            index_r  = 'x;
            onehot_r = 'x;
            for (integer i = N-1; i >= 0; --i) begin
                if (reversed[i]) begin
                    index_r     = LN'(i);
                    onehot_r    = 0;
                    onehot_r[i] = 1'b1;
                end
            end
        end        

        assign index  = index_r;
        assign onehot = onehot_r;
        assign valid_out = (| reversed);

    end    

endmodule
`TRACING_ON
