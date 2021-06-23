`include "VX_platform.vh"

module VX_priority_encoder #( 
    parameter N    = 1,
    parameter FAST = 1,
    parameter LN   = `LOG2UP(N)
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

        assign onehot    = {~data_in[0], data_in[0]};
        assign index     = ~data_in[0];
        assign valid_out = (| data_in);

    end else begin
    
        reg [LN-1:0] index_r;
        reg [N-1:0]  onehot_r;

        if (N == 4)  begin
            always @(*) begin
                casez (data_in)
                4'b???1: begin onehot_r = 4'b0001; index_r = LN'(0); end
                4'b??10: begin onehot_r = 4'b0010; index_r = LN'(1); end
                4'b?100: begin onehot_r = 4'b0100; index_r = LN'(2); end
                4'b1000: begin onehot_r = 4'b1000; index_r = LN'(3); end
                default: begin onehot_r = 'x;      index_r = 'x; end
                endcase
            end
        end else if (N == 8)  begin
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
        end else if (N == 16)  begin
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