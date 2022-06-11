`include "VX_platform.vh"

`TRACING_OFF
module VX_rr_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LOCK_ENABLE  = 0,
    parameter MODEL        = 1,
    localparam LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input  wire                     clk,
    input  wire                     reset,          
    input  wire                     unlock,
    input  wire [NUM_REQS-1:0]      requests, 
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,   
    output wire                     grant_valid
);
    if (NUM_REQS == 1)  begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (unlock)
        
        assign grant_index  = 0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else if (NUM_REQS == 2)  begin

        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;  
        reg [LOG_NUM_REQS-1:0]  state;

        always @(*) begin
            casez ({state, requests})
            3'b0_01,
            3'b1_?1: begin grant_onehot_r = 2'b01; grant_index_r = LOG_NUM_REQS'(0); end
            default: begin grant_onehot_r = 2'b10; grant_index_r = LOG_NUM_REQS'(1); end
            endcase
        end

        always @(posedge clk) begin                       
            if (reset) begin         
                state <= '0;
            end else if (!LOCK_ENABLE || unlock) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);        

    end else if (NUM_REQS == 4)  begin

        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;  
        reg [LOG_NUM_REQS-1:0]  state;

        always @(*) begin
            casez ({state, requests})
            6'b00_0001, 
            6'b01_00?1, 
            6'b10_0??1,
            6'b11_???1: begin grant_onehot_r = 4'b0001; grant_index_r = LOG_NUM_REQS'(0); end
            6'b00_??1?, 
            6'b01_0010, 
            6'b10_0?10, 
            6'b11_??10: begin grant_onehot_r = 4'b0010; grant_index_r = LOG_NUM_REQS'(1); end
            6'b00_?10?, 
            6'b01_?1??, 
            6'b10_0100, 
            6'b11_?100: begin grant_onehot_r = 4'b0100; grant_index_r = LOG_NUM_REQS'(2); end
            default:    begin grant_onehot_r = 4'b1000; grant_index_r = LOG_NUM_REQS'(3); end
            endcase
        end

        always @(posedge clk) begin                       
            if (reset) begin         
                state <= '0;
            end else if (!LOCK_ENABLE || unlock) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);        

    end else if (NUM_REQS == 8)  begin

        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;  
        reg [LOG_NUM_REQS-1:0]  state;

        always @(*) begin
            casez ({state, requests})
            11'b000_00000001, 
            11'b001_000000?1, 
            11'b010_00000??1, 
            11'b011_0000???1,
            11'b100_000????1, 
            11'b101_00?????1, 
            11'b110_0??????1, 
            11'b111_???????1: begin grant_onehot_r = 8'b00000001; grant_index_r = LOG_NUM_REQS'(0); end
            11'b000_??????1?, 
            11'b001_00000010, 
            11'b010_00000?10, 
            11'b011_0000??10,
            11'b100_000???10, 
            11'b101_00????10, 
            11'b110_0?????10, 
            11'b111_??????10: begin grant_onehot_r = 8'b00000010; grant_index_r = LOG_NUM_REQS'(1); end
            11'b000_?????10?, 
            11'b001_?????1??, 
            11'b010_00000100, 
            11'b011_0000?100,
            11'b100_000??100, 
            11'b101_00???100, 
            11'b110_0????100, 
            11'b111_?????100: begin grant_onehot_r = 8'b00000100; grant_index_r = LOG_NUM_REQS'(2); end
            11'b000_????100?,
            11'b001_????10??,
            11'b010_????1???,
            11'b011_00001000,
            11'b100_000?1000,
            11'b101_00??1000,
            11'b110_0???1000,
            11'b111_????1000: begin grant_onehot_r = 8'b00001000; grant_index_r = LOG_NUM_REQS'(3); end
            11'b000_???1000?,
            11'b001_???100??,
            11'b010_???10???,
            11'b011_???1????,
            11'b100_00010000,
            11'b101_00?10000,
            11'b110_0??10000,
            11'b111_???10000: begin grant_onehot_r = 8'b00010000; grant_index_r = LOG_NUM_REQS'(4); end
            11'b000_??10000?,
            11'b001_??1000??,
            11'b010_??100???,
            11'b011_??10????,
            11'b100_??1?????,
            11'b101_00100000,
            11'b110_0?100000,
            11'b111_??100000: begin grant_onehot_r = 8'b00100000; grant_index_r = LOG_NUM_REQS'(5); end
            11'b000_?100000?,
            11'b001_?10000??,
            11'b010_?1000???,
            11'b011_?100????,
            11'b100_?10?????,
            11'b101_?1??????,
            11'b110_01000000,
            11'b111_?1000000: begin grant_onehot_r = 8'b01000000; grant_index_r = LOG_NUM_REQS'(6); end
            default:          begin grant_onehot_r = 8'b10000000; grant_index_r = LOG_NUM_REQS'(7); end
            endcase
        end

        always @(posedge clk) begin                       
            if (reset) begin         
                state <= '0;
            end else if (!LOCK_ENABLE || unlock) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);        
    
    end else if (NUM_REQS == 16)  begin

        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;  
        reg [LOG_NUM_REQS-1:0]  state;

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
    
        always @(posedge clk) begin                       
            if (reset) begin         
                state <= '0;
            end else if (!LOCK_ENABLE || unlock) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);

    end else if (MODEL == 1) begin
    
    `IGNORE_WARNINGS_BEGIN
        wire [NUM_REQS-1:0] mask_higher_pri_regs, unmask_higher_pri_regs;
    `IGNORE_WARNINGS_END
        wire [NUM_REQS-1:0] grant_masked, grant_unmasked;

        reg [NUM_REQS-1:0] pointer_reg;

        wire [NUM_REQS-1:0] req_masked = requests & pointer_reg;

        assign mask_higher_pri_regs[NUM_REQS-1:1] = mask_higher_pri_regs[NUM_REQS-2:0] | req_masked[NUM_REQS-2:0];
        assign mask_higher_pri_regs[0] = 1'b0;
        assign grant_masked[NUM_REQS-1:0] = req_masked[NUM_REQS-1:0] & ~mask_higher_pri_regs[NUM_REQS-1:0];

        assign unmask_higher_pri_regs[NUM_REQS-1:1] = unmask_higher_pri_regs[NUM_REQS-2:0] | requests[NUM_REQS-2:0];
        assign unmask_higher_pri_regs[0] = 1'b0;
        assign grant_unmasked[NUM_REQS-1:0] = requests[NUM_REQS-1:0] & ~unmask_higher_pri_regs[NUM_REQS-1:0];

        wire no_req_masked = ~(|req_masked);
        assign grant_onehot = ({NUM_REQS{no_req_masked}} & grant_unmasked) | grant_masked;

        always @(posedge clk) begin
		    if (reset) begin
				pointer_reg <= {NUM_REQS{1'b1}};
			end else if (!LOCK_ENABLE || unlock) begin
				if (|req_masked) begin
                    pointer_reg <= mask_higher_pri_regs;
                end else if (|requests) begin
                    pointer_reg <= unmask_higher_pri_regs;
                end else begin
                    pointer_reg <= pointer_reg;
                end
			end
	    end

        assign grant_valid = (| requests); 

        VX_onehot_encoder #(
            .N (NUM_REQS)
        ) onehot_encoder (
            .data_in  (grant_onehot),
            .data_out (grant_index),        
            `UNUSED_PIN (valid_out)
        );
    
    end else begin
        
        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;  
        reg [NUM_REQS-1:0]      state;       
        
        always @(*) begin
            grant_index_r  = 'x;
            grant_onehot_r = 'x;
            for (integer i = 0; i < NUM_REQS; ++i) begin
                for (integer j = 0; j < NUM_REQS; ++j) begin
                    if (state[i] && requests[(j + 1) % NUM_REQS]) begin
                        grant_index_r  = LOG_NUM_REQS'((j + 1) % NUM_REQS);
                        grant_onehot_r = '0;
                        grant_onehot_r[(j + 1) % NUM_REQS] = 1;
                    end
                end
            end
        end

        always @(posedge clk) begin                       
            if (reset) begin         
                state <= '0;
            end else if (!LOCK_ENABLE || unlock) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);        
    end
    
endmodule
`TRACING_ON
