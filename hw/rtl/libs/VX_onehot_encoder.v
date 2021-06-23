`include "VX_platform.vh"

// Fast encoder using parallel prefix computation
// Adapter from BaseJump STL: http://bjump.org/data_out.html

module VX_onehot_encoder #(
    parameter N  = 1,
    parameter LN = `LOG2UP(N)
) (
    input wire [N-1:0]   data_in,    
    output wire [LN-1:0] data_out,
    output wire          valid
); 
    if (N == 1) begin

        assign data_out = data_in;
        assign valid    = data_in;

    end else if (N == 2) begin

        assign data_out = data_in[1];
        assign valid    = (| data_in);

    end else begin
    
        reg [LN-1:0] index_r;

        if (N == 4)  begin
            always @(*) begin
                casez (data_in)
                4'b0001: index_r = LN'(0);
                4'b001?: index_r = LN'(1);
                4'b01??: index_r = LN'(2);
                4'b1???: index_r = LN'(3);
                default: index_r = 'x;
                endcase
            end
        end else if (N == 8)  begin
            always @(*) begin
                casez (data_in)
                8'b00000001: index_r = LN'(0);
                8'b0000001?: index_r = LN'(1);
                8'b000001??: index_r = LN'(2);
                8'b00001???: index_r = LN'(3);
                8'b0001????: index_r = LN'(4);
                8'b001?????: index_r = LN'(5);
                8'b01??????: index_r = LN'(6);
                8'b1???????: index_r = LN'(7);
                default:     index_r = 'x;
                endcase
            end
        end else if (N == 16)  begin
            always @(*) begin
                casez (data_in)
                16'b0000000000000001: index_r = LN'(0);
                16'b000000000000001?: index_r = LN'(1);
                16'b00000000000001??: index_r = LN'(2);
                16'b0000000000001???: index_r = LN'(3);
                16'b000000000001????: index_r = LN'(4);
                16'b00000000001?????: index_r = LN'(5);
                16'b0000000001??????: index_r = LN'(6);
                16'b000000001???????: index_r = LN'(7);
                16'b00000001????????: index_r = LN'(8);
                16'b0000001?????????: index_r = LN'(9);
                16'b000001??????????: index_r = LN'(10);
                16'b00001???????????: index_r = LN'(11);
                16'b0001????????????: index_r = LN'(12);
                16'b001?????????????: index_r = LN'(13);
                16'b01??????????????: index_r = LN'(14);
                16'b1???????????????: index_r = LN'(15);
                default:              index_r = 'x;
                endcase
            end
        end else begin        
            always @(*) begin        
                index_r = 'x; 
                for (integer i = 0; i < N; i++) begin
                    if (data_in[i]) begin                
                        index_r = `LOG2UP(N)'(i);
                    end
                end
            end
        end

        assign data_out = index_r;
        assign valid    = (| data_in);

    end

endmodule