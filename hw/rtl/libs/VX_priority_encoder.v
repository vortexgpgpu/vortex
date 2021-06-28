`include "VX_platform.vh"

module VX_priority_encoder #( 
    parameter N       = 1,  
    parameter REVERSE = 0,
    parameter FAST    = 1,
    parameter LN      = `LOG2UP(N)
) (
    input  wire [N-1:0]  data_in,  
    output wire [N-1:0]  onehot,
    output wire [LN-1:0] index,
    output wire          valid_out
);

    if (N == 1) begin

        assign onehot    = data_in;
        assign index     = 0;
        assign valid_out = data_in;

    end else if (N == 2) begin

        assign onehot    = {~data_in[REVERSE], data_in[REVERSE]};
        assign index     = ~data_in[REVERSE];
        assign valid_out = (| data_in);

    end else if (FAST) begin

        wire [N-1:0] scan_lo;

        VX_scan #(
            .N       (N),
            .OP      (2),
            .REVERSE (REVERSE)
        ) scan (
            .data_in  (data_in),
            .data_out (scan_lo)
        );

        if (REVERSE) begin
            assign onehot    = scan_lo & {1'b1, (~scan_lo[N-1:1])};
            assign valid_out = scan_lo[0];
        end else begin
            assign onehot    = scan_lo & {(~scan_lo[N-2:0]), 1'b1};
            assign valid_out = scan_lo[N-1];            
        end

        VX_onehot_encoder #(
            .N (N),
            .REVERSE (REVERSE)
        ) onehot_encoder (
            .data_in  (onehot),
            .data_out (index),        
            `UNUSED_PIN (valid)
        );

    end else begin

        reg [LN-1:0] index_r;
        reg [N-1:0]  onehot_r;

        if (REVERSE) begin            
            always @(*) begin
                index_r  = 'x;
                onehot_r = 'x;
                for (integer i = 0; i < N; ++i) begin
                    if (data_in[i]) begin
                        index_r     = LN'(i);
                        onehot_r    = 0;
                        onehot_r[i] = 1'b1;
                    end
                end
            end
        end else begin
            always @(*) begin
                index_r  = 'x;
                onehot_r = 'x;
                for (integer i = N-1; i >= 0; --i) begin
                    if (data_in[i]) begin
                        index_r     = LN'(i);
                        onehot_r    = 0;
                        onehot_r[i] = 1'b1;
                    end
                end
            end
        end        

        assign index  = index_r;
        assign onehot = onehot_r;
        assign valid_out = (| data_in);        

    end    

endmodule