/* Leading One Detector for OP TCU */

`include "VX_define.vh"

module VX_tcu_lod4 # (
    parameter ONE_HOT = 0
) (
    input  wire [3:0] in,
    output wire [(ONE_HOT ? 3 : 2):0] out,      // one-hot or index of position of MSb '1' 
    output wire [3:0] next_stage
);
    generate
        if (ONE_HOT) begin : g_one_hot
            assign out[3] =  in[3];
            assign out[2] = ~in[3] &  in[2];
            assign out[1] = ~in[3] & ~in[2] &  in[1];
            assign out[0] = ~in[3] & ~in[2] & ~in[1] & in[0];

            assign next_stage = in & ~out;  // Pass to the next LOD the bitmap except for the place it finds
        
        end else begin : g_index

            assign out = in[3] ? 3'd4 :
                         in[2] ? 3'd3 :
                         in[1] ? 3'd2 :
                         in[0] ? 3'd1 :
                                 3'd0;     // no 1's
                                
            assign next_stage = in[3] ? in & 4'b0111 :
                                in[2] ? in & 4'b0011 :
                                in[1] ? in & 4'b0001 :
                                in[0] ?      4'b0000 :
                                             4'b0000;   // no 1's

            // always @* begin
            //     casez (in)
            //         4'b1???: begin
            //             out = 3'd4;  // bit 3
            //             next_stage = in & 4'b0111;
            //         end
            //         4'b01??: begin
            //             out = 3'd3;  // bit 2
            //             next_stage = in & 4'b0011;
            //         end
            //         4'b001?: begin
            //             out = 3'd2;  // bit 1
            //             next_stage = in & 4'b0001;
            //         end
            //         4'b0001: begin
            //             out = 3'd1;  // bit 0
            //             next_stage = 4'b0000;
            //         end
            //         default: begin
            //             out = 3'd0;  // no 1s
            //             next_stage = 4'b0000;
            //         end
            //     endcase
            // end
        end
    endgenerate

endmodule
