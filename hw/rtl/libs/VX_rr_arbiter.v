`include "VX_platform.vh"

module VX_rr_arbiter #(
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
        
        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;  
        reg [LOG_NUM_REQS-1:0]  state;

        if (NUM_REQS == 2)  begin
            always @(*) begin
                casez ({state, requests})
                3'b0_1?: begin grant_onehot_r = 2'b10; grant_index_r = LOG_NUM_REQS'(1); end
                3'b0_01: begin grant_onehot_r = 2'b01; grant_index_r = LOG_NUM_REQS'(0); end
                3'b1_?1: begin grant_onehot_r = 2'b01; grant_index_r = LOG_NUM_REQS'(0); end
                3'b1_10: begin grant_onehot_r = 2'b10; grant_index_r = LOG_NUM_REQS'(1); end
                default: begin grant_onehot_r = 'x;    grant_index_r = 'x; end
                endcase
            end
        end else if (NUM_REQS == 4)  begin
            always @(*) begin
                casez ({state, requests})
                6'b00_??1?: begin grant_onehot_r = 4'b0010; grant_index_r = LOG_NUM_REQS'(1); end
                6'b00_?10?: begin grant_onehot_r = 4'b0100; grant_index_r = LOG_NUM_REQS'(2); end
                6'b00_100?: begin grant_onehot_r = 4'b1000; grant_index_r = LOG_NUM_REQS'(3); end
                6'b00_0001: begin grant_onehot_r = 4'b0001; grant_index_r = LOG_NUM_REQS'(0); end
                6'b01_?1??: begin grant_onehot_r = 4'b0100; grant_index_r = LOG_NUM_REQS'(2); end
                6'b01_10??: begin grant_onehot_r = 4'b1000; grant_index_r = LOG_NUM_REQS'(3); end
                6'b01_00?1: begin grant_onehot_r = 4'b0001; grant_index_r = LOG_NUM_REQS'(0); end
                6'b01_0010: begin grant_onehot_r = 4'b0010; grant_index_r = LOG_NUM_REQS'(1); end
                6'b10_1???: begin grant_onehot_r = 4'b1000; grant_index_r = LOG_NUM_REQS'(3); end
                6'b10_0??1: begin grant_onehot_r = 4'b0001; grant_index_r = LOG_NUM_REQS'(0); end
                6'b10_0?10: begin grant_onehot_r = 4'b0010; grant_index_r = LOG_NUM_REQS'(1); end
                6'b10_0100: begin grant_onehot_r = 4'b0100; grant_index_r = LOG_NUM_REQS'(2); end
                6'b11_???1: begin grant_onehot_r = 4'b0001; grant_index_r = LOG_NUM_REQS'(0); end
                6'b11_??10: begin grant_onehot_r = 4'b0010; grant_index_r = LOG_NUM_REQS'(1); end
                6'b11_?100: begin grant_onehot_r = 4'b0100; grant_index_r = LOG_NUM_REQS'(2); end
                6'b11_1000: begin grant_onehot_r = 4'b1000; grant_index_r = LOG_NUM_REQS'(3); end
                default:    begin grant_onehot_r = 'x;      grant_index_r = 'x; end
                endcase
            end
        end else if (NUM_REQS == 8)  begin
            always @(*) begin
                casez ({state, requests})
                11'b000_??????1?: begin grant_onehot_r = 8'b00000010; grant_index_r = LOG_NUM_REQS'(1); end
                11'b000_?????10?: begin grant_onehot_r = 8'b00000100; grant_index_r = LOG_NUM_REQS'(2); end
                11'b000_????100?: begin grant_onehot_r = 8'b00001000; grant_index_r = LOG_NUM_REQS'(3); end
                11'b000_???1000?: begin grant_onehot_r = 8'b00010000; grant_index_r = LOG_NUM_REQS'(4); end
                11'b000_??10000?: begin grant_onehot_r = 8'b00100000; grant_index_r = LOG_NUM_REQS'(5); end
                11'b000_?100000?: begin grant_onehot_r = 8'b01000000; grant_index_r = LOG_NUM_REQS'(6); end
                11'b000_1000000?: begin grant_onehot_r = 8'b10000000; grant_index_r = LOG_NUM_REQS'(7); end
                11'b000_00000001: begin grant_onehot_r = 8'b00000001; grant_index_r = LOG_NUM_REQS'(0); end
                11'b001_?????1??: begin grant_onehot_r = 8'b00000100; grant_index_r = LOG_NUM_REQS'(2); end
                11'b001_????10??: begin grant_onehot_r = 8'b00001000; grant_index_r = LOG_NUM_REQS'(3); end
                11'b001_???100??: begin grant_onehot_r = 8'b00010000; grant_index_r = LOG_NUM_REQS'(4); end
                11'b001_??1000??: begin grant_onehot_r = 8'b00100000; grant_index_r = LOG_NUM_REQS'(5); end
                11'b001_?10000??: begin grant_onehot_r = 8'b01000000; grant_index_r = LOG_NUM_REQS'(6); end
                11'b001_100000??: begin grant_onehot_r = 8'b10000000; grant_index_r = LOG_NUM_REQS'(7); end
                11'b001_000000?1: begin grant_onehot_r = 8'b00000001; grant_index_r = LOG_NUM_REQS'(0); end
                11'b001_00000010: begin grant_onehot_r = 8'b00000010; grant_index_r = LOG_NUM_REQS'(1); end
                11'b010_????1???: begin grant_onehot_r = 8'b00001000; grant_index_r = LOG_NUM_REQS'(3); end
                11'b010_???10???: begin grant_onehot_r = 8'b00010000; grant_index_r = LOG_NUM_REQS'(4); end
                11'b010_??100???: begin grant_onehot_r = 8'b00100000; grant_index_r = LOG_NUM_REQS'(5); end
                11'b010_?1000???: begin grant_onehot_r = 8'b01000000; grant_index_r = LOG_NUM_REQS'(6); end
                11'b010_10000???: begin grant_onehot_r = 8'b10000000; grant_index_r = LOG_NUM_REQS'(7); end
                11'b010_00000??1: begin grant_onehot_r = 8'b00000001; grant_index_r = LOG_NUM_REQS'(0); end
                11'b010_00000?10: begin grant_onehot_r = 8'b00000010; grant_index_r = LOG_NUM_REQS'(1); end
                11'b010_00000100: begin grant_onehot_r = 8'b00000100; grant_index_r = LOG_NUM_REQS'(2); end
                11'b011_???1????: begin grant_onehot_r = 8'b00010000; grant_index_r = LOG_NUM_REQS'(4); end
                11'b011_??10????: begin grant_onehot_r = 8'b00100000; grant_index_r = LOG_NUM_REQS'(5); end
                11'b011_?100????: begin grant_onehot_r = 8'b01000000; grant_index_r = LOG_NUM_REQS'(6); end
                11'b011_1000????: begin grant_onehot_r = 8'b10000000; grant_index_r = LOG_NUM_REQS'(7); end
                11'b011_0000???1: begin grant_onehot_r = 8'b00000001; grant_index_r = LOG_NUM_REQS'(0); end
                11'b011_0000??10: begin grant_onehot_r = 8'b00000010; grant_index_r = LOG_NUM_REQS'(1); end
                11'b011_0000?100: begin grant_onehot_r = 8'b00000100; grant_index_r = LOG_NUM_REQS'(2); end
                11'b011_00001000: begin grant_onehot_r = 8'b00001000; grant_index_r = LOG_NUM_REQS'(3); end
                11'b100_??1?????: begin grant_onehot_r = 8'b00100000; grant_index_r = LOG_NUM_REQS'(5); end
                11'b100_?10?????: begin grant_onehot_r = 8'b01000000; grant_index_r = LOG_NUM_REQS'(6); end
                11'b100_100?????: begin grant_onehot_r = 8'b10000000; grant_index_r = LOG_NUM_REQS'(7); end
                11'b100_000????1: begin grant_onehot_r = 8'b00000001; grant_index_r = LOG_NUM_REQS'(0); end
                11'b100_000???10: begin grant_onehot_r = 8'b00000010; grant_index_r = LOG_NUM_REQS'(1); end
                11'b100_000??100: begin grant_onehot_r = 8'b00000100; grant_index_r = LOG_NUM_REQS'(2); end
                11'b100_000?1000: begin grant_onehot_r = 8'b00001000; grant_index_r = LOG_NUM_REQS'(3); end
                11'b100_00010000: begin grant_onehot_r = 8'b00010000; grant_index_r = LOG_NUM_REQS'(4); end
                11'b101_?1??????: begin grant_onehot_r = 8'b01000000; grant_index_r = LOG_NUM_REQS'(6); end
                11'b101_10??????: begin grant_onehot_r = 8'b10000000; grant_index_r = LOG_NUM_REQS'(7); end
                11'b101_00?????1: begin grant_onehot_r = 8'b00000001; grant_index_r = LOG_NUM_REQS'(0); end
                11'b101_00????10: begin grant_onehot_r = 8'b00000010; grant_index_r = LOG_NUM_REQS'(1); end
                11'b101_00???100: begin grant_onehot_r = 8'b00000100; grant_index_r = LOG_NUM_REQS'(2); end
                11'b101_00??1000: begin grant_onehot_r = 8'b00001000; grant_index_r = LOG_NUM_REQS'(3); end
                11'b101_00?10000: begin grant_onehot_r = 8'b00010000; grant_index_r = LOG_NUM_REQS'(4); end
                11'b101_00100000: begin grant_onehot_r = 8'b00100000; grant_index_r = LOG_NUM_REQS'(5); end
                11'b110_1???????: begin grant_onehot_r = 8'b10000000; grant_index_r = LOG_NUM_REQS'(7); end
                11'b110_0??????1: begin grant_onehot_r = 8'b00000001; grant_index_r = LOG_NUM_REQS'(0); end
                11'b110_0?????10: begin grant_onehot_r = 8'b00000010; grant_index_r = LOG_NUM_REQS'(1); end
                11'b110_0????100: begin grant_onehot_r = 8'b00000100; grant_index_r = LOG_NUM_REQS'(2); end
                11'b110_0???1000: begin grant_onehot_r = 8'b00001000; grant_index_r = LOG_NUM_REQS'(3); end
                11'b110_0??10000: begin grant_onehot_r = 8'b00010000; grant_index_r = LOG_NUM_REQS'(4); end
                11'b110_0?100000: begin grant_onehot_r = 8'b00100000; grant_index_r = LOG_NUM_REQS'(5); end
                11'b110_01000000: begin grant_onehot_r = 8'b01000000; grant_index_r = LOG_NUM_REQS'(6); end
                11'b111_???????1: begin grant_onehot_r = 8'b00000001; grant_index_r = LOG_NUM_REQS'(0); end
                11'b111_??????10: begin grant_onehot_r = 8'b00000010; grant_index_r = LOG_NUM_REQS'(1); end
                11'b111_?????100: begin grant_onehot_r = 8'b00000100; grant_index_r = LOG_NUM_REQS'(2); end
                11'b111_????1000: begin grant_onehot_r = 8'b00001000; grant_index_r = LOG_NUM_REQS'(3); end
                11'b111_???10000: begin grant_onehot_r = 8'b00010000; grant_index_r = LOG_NUM_REQS'(4); end
                11'b111_??100000: begin grant_onehot_r = 8'b00100000; grant_index_r = LOG_NUM_REQS'(5); end
                11'b111_?1000000: begin grant_onehot_r = 8'b01000000; grant_index_r = LOG_NUM_REQS'(6); end
                11'b111_10000000: begin grant_onehot_r = 8'b10000000; grant_index_r = LOG_NUM_REQS'(7); end
                default:          begin grant_onehot_r = 'x;          grant_index_r = 'x; end
                endcase
            end
        end else if (NUM_REQS == 16)  begin
            always @(*) begin
                casez ({state, requests})
                20'b0000_??????????????1?: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b0000_?????????????10?: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b0000_????????????100?: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b0000_???????????1000?: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b0000_??????????10000?: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b0000_?????????100000?: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b0000_????????1000000?: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b0000_???????10000000?: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b0000_??????100000000?: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b0000_?????1000000000?: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b0000_????10000000000?: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b0000_???100000000000?: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b0000_??1000000000000?: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b0000_?10000000000000?: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b0000_100000000000000?: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b0000_0000000000000001: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b0001_?????????????1??: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b0001_????????????10??: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b0001_???????????100??: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b0001_??????????1000??: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b0001_?????????10000??: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b0001_????????100000??: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b0001_???????1000000??: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b0001_??????10000000??: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b0001_?????100000000??: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b0001_????1000000000??: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b0001_???10000000000??: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b0001_??100000000000??: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b0001_?1000000000000??: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b0001_10000000000000??: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b0001_00000000000000?1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b0001_0000000000000010: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b0010_????????????1???: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b0010_???????????10???: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b0010_??????????100???: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b0010_?????????1000???: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b0010_????????10000???: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b0010_???????100000???: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b0010_??????1000000???: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b0010_?????10000000???: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b0010_????100000000???: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b0010_???1000000000???: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b0010_??10000000000???: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b0010_?100000000000???: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b0010_1000000000000???: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b0010_0000000000000??1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b0010_0000000000000?10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b0010_0000000000000100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b0011_???????????1????: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b0011_??????????10????: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b0011_?????????100????: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b0011_????????1000????: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b0011_???????10000????: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b0011_??????100000????: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b0011_?????1000000????: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b0011_????10000000????: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b0011_???100000000????: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b0011_??1000000000????: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b0011_?10000000000????: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b0011_100000000000????: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b0011_000000000000???1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b0011_000000000000??10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b0011_000000000000?100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b0011_0000000000001000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b0100_??????????1?????: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b0100_?????????10?????: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b0100_????????100?????: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b0100_???????1000?????: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b0100_??????10000?????: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b0100_?????100000?????: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b0100_????1000000?????: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b0100_???10000000?????: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b0100_??100000000?????: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b0100_?1000000000?????: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b0100_10000000000?????: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b0100_00000000000????1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b0100_00000000000???10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b0100_00000000000??100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b0100_00000000000?1000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b0100_0000000000010000: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b0101_?????????1??????: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b0101_????????10??????: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b0101_???????100??????: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b0101_??????1000??????: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b0101_?????10000??????: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b0101_????100000??????: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b0101_???1000000??????: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b0101_??10000000??????: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b0101_?100000000??????: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b0101_1000000000??????: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b0101_0000000000?????1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b0101_0000000000????10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b0101_0000000000???100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b0101_0000000000??1000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b0101_0000000000?10000: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b0101_0000000000100000: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b0110_????????1???????: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b0110_???????10???????: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b0110_??????100???????: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b0110_?????1000???????: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b0110_????10000???????: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b0110_???100000???????: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b0110_??1000000???????: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b0110_?10000000???????: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b0110_100000000???????: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b0110_000000000??????1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b0110_000000000?????10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b0110_000000000????100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b0110_000000000???1000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b0110_000000000??10000: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b0110_000000000?100000: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b0110_0000000001000000: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b0111_???????1????????: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b0111_??????10????????: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b0111_?????100????????: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b0111_????1000????????: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b0111_???10000????????: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b0111_??100000????????: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b0111_?1000000????????: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b0111_10000000????????: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b0111_00000000???????1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b0111_00000000??????10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b0111_00000000?????100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b0111_00000000????1000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b0111_00000000???10000: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b0111_00000000??100000: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b0111_00000000?1000000: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b0111_0000000010000000: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b1000_??????1?????????: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b1000_?????10?????????: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b1000_????100?????????: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b1000_???1000?????????: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b1000_??10000?????????: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b1000_?100000?????????: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b1000_1000000?????????: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b1000_0000000????????1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b1000_0000000???????10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b1000_0000000??????100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b1000_0000000?????1000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b1000_0000000????10000: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b1000_0000000???100000: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b1000_0000000??1000000: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b1000_0000000?10000000: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b1000_0000000100000000: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b1001_?????1??????????: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b1001_????10??????????: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b1001_???100??????????: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b1001_??1000??????????: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b1001_?10000??????????: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b1001_100000??????????: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b1001_000000?????????1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b1001_000000????????10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b1001_000000???????100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b1001_000000??????1000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b1001_000000?????10000: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b1001_000000????100000: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b1001_000000???1000000: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b1001_000000??10000000: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b1001_000000?100000000: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b1001_0000001000000000: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b1010_????1???????????: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b1010_???10???????????: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b1010_??100???????????: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b1010_?1000???????????: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b1010_10000???????????: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b1010_00000??????????1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b1010_00000?????????10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b1010_00000????????100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b1010_00000???????1000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b1010_00000??????10000: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b1010_00000?????100000: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b1010_00000????1000000: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b1010_00000???10000000: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b1010_00000??100000000: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b1010_00000?1000000000: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b1010_0000010000000000: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b1011_???1????????????: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b1011_??10????????????: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b1011_?100????????????: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b1011_1000????????????: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b1011_0000???????????1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b1011_0000??????????10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b1011_0000?????????100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b1011_0000????????1000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b1011_0000???????10000: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b1011_0000??????100000: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b1011_0000?????1000000: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b1011_0000????10000000: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b1011_0000???100000000: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b1011_0000??1000000000: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b1011_0000?10000000000: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b1011_0000100000000000: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b1100_??1?????????????: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b1100_?10?????????????: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b1100_100?????????????: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b1100_000????????????1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b1100_000???????????10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b1100_000??????????100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b1100_000?????????1000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b1100_000????????10000: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b1100_000???????100000: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b1100_000??????1000000: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b1100_000?????10000000: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b1100_000????100000000: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b1100_000???1000000000: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b1100_000??10000000000: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b1100_000?100000000000: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b1100_0001000000000000: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b1101_?1??????????????: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b1101_10??????????????: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b1101_00?????????????1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b1101_00????????????10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b1101_00???????????100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b1101_00??????????1000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b1101_00?????????10000: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b1101_00????????100000: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b1101_00???????1000000: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b1101_00??????10000000: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b1101_00?????100000000: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b1101_00????1000000000: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b1101_00???10000000000: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b1101_00??100000000000: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b1101_00?1000000000000: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b1101_0010000000000000: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b1110_1???????????????: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                20'b1110_0??????????????1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b1110_0?????????????10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b1110_0????????????100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b1110_0???????????1000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b1110_0??????????10000: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b1110_0?????????100000: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b1110_0????????1000000: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b1110_0???????10000000: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b1110_0??????100000000: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b1110_0?????1000000000: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b1110_0????10000000000: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b1110_0???100000000000: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b1110_0??1000000000000: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b1110_0?10000000000000: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b1110_0100000000000000: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b1111_???????????????1: begin grant_onehot_r = 16'b0000000000000001; grant_index_r = LOG_NUM_REQS'(0); end
                20'b1111_??????????????10: begin grant_onehot_r = 16'b0000000000000010; grant_index_r = LOG_NUM_REQS'(1); end
                20'b1111_?????????????100: begin grant_onehot_r = 16'b0000000000000100; grant_index_r = LOG_NUM_REQS'(2); end
                20'b1111_????????????1000: begin grant_onehot_r = 16'b0000000000001000; grant_index_r = LOG_NUM_REQS'(3); end
                20'b1111_???????????10000: begin grant_onehot_r = 16'b0000000000010000; grant_index_r = LOG_NUM_REQS'(4); end
                20'b1111_??????????100000: begin grant_onehot_r = 16'b0000000000100000; grant_index_r = LOG_NUM_REQS'(5); end
                20'b1111_?????????1000000: begin grant_onehot_r = 16'b0000000001000000; grant_index_r = LOG_NUM_REQS'(6); end
                20'b1111_????????10000000: begin grant_onehot_r = 16'b0000000010000000; grant_index_r = LOG_NUM_REQS'(7); end
                20'b1111_???????100000000: begin grant_onehot_r = 16'b0000000100000000; grant_index_r = LOG_NUM_REQS'(8); end
                20'b1111_??????1000000000: begin grant_onehot_r = 16'b0000001000000000; grant_index_r = LOG_NUM_REQS'(9); end
                20'b1111_?????10000000000: begin grant_onehot_r = 16'b0000010000000000; grant_index_r = LOG_NUM_REQS'(10); end
                20'b1111_????100000000000: begin grant_onehot_r = 16'b0000100000000000; grant_index_r = LOG_NUM_REQS'(11); end
                20'b1111_???1000000000000: begin grant_onehot_r = 16'b0001000000000000; grant_index_r = LOG_NUM_REQS'(12); end
                20'b1111_??10000000000000: begin grant_onehot_r = 16'b0010000000000000; grant_index_r = LOG_NUM_REQS'(13); end
                20'b1111_?100000000000000: begin grant_onehot_r = 16'b0100000000000000; grant_index_r = LOG_NUM_REQS'(14); end
                20'b1111_1000000000000000: begin grant_onehot_r = 16'b1000000000000000; grant_index_r = LOG_NUM_REQS'(15); end
                default:                   begin grant_onehot_r = 'x;                   grant_index_r = 'x; end
                endcase
            end
        end else begin
            always @(*) begin
                grant_index_r  = 'x;
                grant_onehot_r = 'x;
                for (integer i = 0; i < NUM_REQS; ++i) begin    
                    for (integer j = 0; j < NUM_REQS; ++j) begin
                        if (state == LOG_NUM_REQS'(i)
                         && requests[(j + 1) % NUM_REQS]) begin                        
                            grant_index_r  = LOG_NUM_REQS'((j + 1) % NUM_REQS);
                            grant_onehot_r = '0;
                            grant_onehot_r[(j + 1) % NUM_REQS] = 1;
                        end
                    end
                end
            end
        end     

        always @(posedge clk) begin                       
            if (reset) begin         
                state <= 0;
            end else if (!LOCK_ENABLE || enable) begin
                state <= grant_index;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);        
    end
    
endmodule