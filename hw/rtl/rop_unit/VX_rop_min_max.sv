`include "VX_rop_define.vh"

module VX_rop_min_max #(  
    parameter CORE_ID = 0
) (
    input wire [31:0] src_color,
    input wire [31:0] dst_color,

    output wire [31:0] min_color_out,
    output wire [31:0] max_color_out
);

    always @(*) begin
        // Compare Red Components
        if(src_color[31:24] > dst_color[31:24]) begin
            max_color_out[31:24] = src_color[31:24];
            min_color_out[31:24] = dst_color[31:24];
        end
        else begin
            max_color_out[31:24] = dst_color[31:24];
            min_color_out[31:24] = src_color[31:24];
        end
        // Compare Green Components
        if(src_color[23:16] > dst_color[23:16]) begin
            max_color_out[23:16] = src_color[23:16];
            min_color_out[23:16] = dst_color[23:16];
        end
        else begin
            max_color_out[23:16] = dst_color[23:16];
            min_color_out[23:16] = src_color[23:16];
        end
        // Compare Blue Components
        if(src_color[15:8] > dst_color[15:8]) begin
            max_color_out[15:8] = src_color[15:8];
            min_color_out[15:8] = dst_color[15:8];
        end
        else begin
            max_color_out[15:8] = dst_color[15:8];
            min_color_out[15:8] = src_color[15:8];
        end
        // Compare Alpha Components
        if(src_color[7:0] > dst_color[7:0]) begin
            max_color_out[7:0] = src_color[7:0];
            min_color_out[7:0] = dst_color[7:0];
        end
        else begin
            max_color_out[7:0] = dst_color[7:0];
            min_color_out[7:0] = src_color[7:0];
        end
    end


endmodule