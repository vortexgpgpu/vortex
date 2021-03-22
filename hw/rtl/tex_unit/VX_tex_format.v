`include "VX_tex_define.vh"

module VX_tex_format #(
    parameter CORE_ID = 0,
    parameter NUM_TEXELS = 4 //BILINEAR
) (
    input wire [NUM_TEXELS-1:0][31:0]    texel_data,
    input wire [`TEX_FORMAT_BITS-1:0]    format,

    output wire [`NUM_COLOR_CHANNEL-1:0] color_enable,
    output wire [NUM_TEXELS-1:0][63:0]   formatted_texel     
);  
    `UNUSED_PARAM (CORE_ID)

    reg [`NUM_COLOR_CHANNEL-1:0] color_enable_r;
    reg [NUM_TEXELS][63:0]       formatted_texel_r;    

    always @(*) begin
        for (integer i = 0; i<NUM_TEXELS ;i++ ) begin
            case (format)
                `TEX_FORMAT_R5G6B5: begin
                    formatted_texel_r[i][55:48] = `TEX_COLOR_BITS'(texel_data[i][15:11]);
                    formatted_texel_r[i][39:32] = `TEX_COLOR_BITS'(texel_data[i][10:5]);
                    formatted_texel_r[i][23:16] = `TEX_COLOR_BITS'(texel_data[i][4:0]);
                    formatted_texel_r[i][7:0]   = {`TEX_COLOR_BITS{1'b0}};
                    if (i == 0)
                        color_enable_r = 4'b1110;
                end
                `TEX_FORMAT_R8G8B8: begin
                    formatted_texel_r[i][55:48] = `TEX_COLOR_BITS'(texel_data[i][23:16]);
                    formatted_texel_r[i][39:32] = `TEX_COLOR_BITS'(texel_data[i][15:8]);
                    formatted_texel_r[i][23:16] = `TEX_COLOR_BITS'(texel_data[i][7:0]);
                    formatted_texel_r[i][7:0]   = {`TEX_COLOR_BITS{1'b0}};
                    if (i == 0)
                        color_enable_r = 4'b1110;
                end
                default: begin // `TEX_FORMAT_R8G8B8A8:
                    formatted_texel_r[i][55:48] = `TEX_COLOR_BITS'(texel_data[i][31:24]);
                    formatted_texel_r[i][39:32] = `TEX_COLOR_BITS'(texel_data[i][23:16]);
                    formatted_texel_r[i][23:16] = `TEX_COLOR_BITS'(texel_data[i][15:8]);
                    formatted_texel_r[i][7:0]   = `TEX_COLOR_BITS'(texel_data[i][7:0]);
                    if (i == 0)
                        color_enable_r = 4'b1111;
                end
            endcase
        end
    end

    assign color_enable    = color_enable_r;
    assign formatted_texel = formatted_texel_r & 64'h00ff00ff00ff00ff;

endmodule
