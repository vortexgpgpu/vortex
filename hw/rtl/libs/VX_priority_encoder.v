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

    end else if (N == 4)  begin
            
        reg [LN-1:0] index_r;
        reg [N-1:0]  onehot_r;

        if (REVERSE) begin
            always @(*) begin
                casez (data_in)
                4'b1???: begin onehot_r = 4'b0001; index_r = LN'(0); end
                4'b01??: begin onehot_r = 4'b0010; index_r = LN'(1); end
                4'b001?: begin onehot_r = 4'b0100; index_r = LN'(2); end
                4'b0001: begin onehot_r = 4'b1000; index_r = LN'(3); end
                default: begin onehot_r = 'x;      index_r = 'x; end
                endcase
            end
        end else begin
            always @(*) begin
                casez (data_in)
                4'b???1: begin onehot_r = 4'b0001; index_r = LN'(0); end
                4'b??10: begin onehot_r = 4'b0010; index_r = LN'(1); end
                4'b?100: begin onehot_r = 4'b0100; index_r = LN'(2); end
                4'b1000: begin onehot_r = 4'b1000; index_r = LN'(3); end
                default: begin onehot_r = 'x;      index_r = 'x; end
                endcase
            end
        end

        assign index  = index_r;
        assign onehot = onehot_r;

    end else if (N == 8)  begin
        
        reg [LN-1:0] index_r;
        reg [N-1:0]  onehot_r;

        if (REVERSE) begin
            always @(*) begin
                casez (data_in)
                8'b1???????: begin onehot_r = 8'b00000001; index_r = LN'(0); end
                8'b01??????: begin onehot_r = 8'b00000010; index_r = LN'(1); end
                8'b001?????: begin onehot_r = 8'b00000100; index_r = LN'(2); end
                8'b0001????: begin onehot_r = 8'b00001000; index_r = LN'(3); end
                8'b00001???: begin onehot_r = 8'b00010000; index_r = LN'(4); end
                8'b000001??: begin onehot_r = 8'b00100000; index_r = LN'(5); end
                8'b0000001?: begin onehot_r = 8'b01000000; index_r = LN'(6); end
                8'b00000001: begin onehot_r = 8'b10000000; index_r = LN'(7); end
                default:     begin onehot_r = 'x;          index_r = 'x; end
                endcase
            end
        end else begin
            always @(*) begin
                casez (data_in)
                8'b???????1: begin onehot_r = 8'b00000001; index_r = LN'(0); end
                8'b??????10: begin onehot_r = 8'b00000010; index_r = LN'(1); end
                8'b?????100: begin onehot_r = 8'b00000100; index_r = LN'(2); end
                8'b????1000: begin onehot_r = 8'b00001000; index_r = LN'(3); end
                8'b???10000: begin onehot_r = 8'b00010000; index_r = LN'(4); end
                8'b??100000: begin onehot_r = 8'b00100000; index_r = LN'(5); end
                8'b?1000000: begin onehot_r = 8'b01000000; index_r = LN'(6); end
                8'b10000000: begin onehot_r = 8'b10000000; index_r = LN'(7); end
                default:     begin onehot_r = 'x;          index_r = 'x; end
                endcase
            end
        end

        assign index  = index_r;
        assign onehot = onehot_r;

    end else if (N == 16)  begin
        
        reg [LN-1:0] index_r;
        reg [N-1:0]  onehot_r;

        if (REVERSE) begin
            always @(*) begin
                casez (data_in)
                16'b1???????????????: begin onehot_r = 16'b0000000000000001; index_r = LN'(0); end
                16'b01??????????????: begin onehot_r = 16'b0000000000000010; index_r = LN'(1); end
                16'b001?????????????: begin onehot_r = 16'b0000000000000100; index_r = LN'(2); end
                16'b0001????????????: begin onehot_r = 16'b0000000000001000; index_r = LN'(3); end
                16'b00001???????????: begin onehot_r = 16'b0000000000010000; index_r = LN'(4); end
                16'b000001??????????: begin onehot_r = 16'b0000000000100000; index_r = LN'(5); end
                16'b0000001?????????: begin onehot_r = 16'b0000000001000000; index_r = LN'(6); end
                16'b00000001????????: begin onehot_r = 16'b0000000010000000; index_r = LN'(7); end
                16'b000000001???????: begin onehot_r = 16'b0000000100000000; index_r = LN'(8); end
                16'b0000000001??????: begin onehot_r = 16'b0000001000000000; index_r = LN'(9); end
                16'b00000000001?????: begin onehot_r = 16'b0000010000000000; index_r = LN'(10); end
                16'b000000000001????: begin onehot_r = 16'b0000100000000000; index_r = LN'(11); end
                16'b0000000000001???: begin onehot_r = 16'b0001000000000000; index_r = LN'(12); end
                16'b00000000000001??: begin onehot_r = 16'b0010000000000000; index_r = LN'(13); end
                16'b000000000000001?: begin onehot_r = 16'b0100000000000000; index_r = LN'(14); end
                16'b0000000000000001: begin onehot_r = 16'b1000000000000000; index_r = LN'(15); end
                default:              begin onehot_r = 'x;                   index_r = 'x; end
                endcase
            end
        end else begin
            always @(*) begin
                casez (data_in)
                16'b???????????????1: begin onehot_r = 16'b0000000000000001; index_r = LN'(0); end
                16'b??????????????10: begin onehot_r = 16'b0000000000000010; index_r = LN'(1); end
                16'b?????????????100: begin onehot_r = 16'b0000000000000100; index_r = LN'(2); end
                16'b????????????1000: begin onehot_r = 16'b0000000000001000; index_r = LN'(3); end
                16'b???????????10000: begin onehot_r = 16'b0000000000010000; index_r = LN'(4); end
                16'b??????????100000: begin onehot_r = 16'b0000000000100000; index_r = LN'(5); end
                16'b?????????1000000: begin onehot_r = 16'b0000000001000000; index_r = LN'(6); end
                16'b????????10000000: begin onehot_r = 16'b0000000010000000; index_r = LN'(7); end
                16'b???????100000000: begin onehot_r = 16'b0000000100000000; index_r = LN'(8); end
                16'b??????1000000000: begin onehot_r = 16'b0000001000000000; index_r = LN'(9); end
                16'b?????10000000000: begin onehot_r = 16'b0000010000000000; index_r = LN'(10); end
                16'b????100000000000: begin onehot_r = 16'b0000100000000000; index_r = LN'(11); end
                16'b???1000000000000: begin onehot_r = 16'b0001000000000000; index_r = LN'(12); end
                16'b??10000000000000: begin onehot_r = 16'b0010000000000000; index_r = LN'(13); end
                16'b?100000000000000: begin onehot_r = 16'b0100000000000000; index_r = LN'(14); end
                16'b1000000000000000: begin onehot_r = 16'b1000000000000000; index_r = LN'(15); end
                default:              begin onehot_r = 'x;                   index_r = 'x; end
                endcase
            end
        end

        assign index  = index_r;
        assign onehot = onehot_r;

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

    end    

    assign valid_out = (| data_in);        
    
endmodule