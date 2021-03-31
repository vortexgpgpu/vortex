`include "VX_tex_define.vh"

module VX_tex_format #(
    parameter CORE_ID    = 0,
    parameter NUM_TEXELS = 4 //BILINEAR
) (
    input wire [NUM_TEXELS-1:0][31:0]    texel_data,
    input wire [`TEX_FORMAT_BITS-1:0]    format,

    output wire [`NUM_COLOR_CHANNEL-1:0] color_enable,
    output wire [NUM_TEXELS-1:0][63:0]   formatted_lerp_texel,     
    output wire [31:0]                   formatted_pt_texel
);  
    `UNUSED_PARAM (CORE_ID)

    reg [`NUM_COLOR_CHANNEL-1:0] color_enable_r;
    reg [NUM_TEXELS-1:0][63:0] formatted_texel_r;   
    reg [31:0]                 formatted_pt_r; 

    always @(*) begin
        // bilerp/trilerp input
        for (integer i = 0; i<NUM_TEXELS ;i++ ) begin
            case (format)
                `TEX_FORMAT_R5G6B5: begin
                    formatted_texel_r[i][07:00] = `TEX_COLOR_BITS'(texel_data[i][4:0]);
                    formatted_texel_r[i][23:16] = `TEX_COLOR_BITS'(texel_data[i][10:5]);
                    formatted_texel_r[i][39:32] = `TEX_COLOR_BITS'(texel_data[i][15:11]);
                    formatted_texel_r[i][55:48] = {`TEX_COLOR_BITS{1'b0}};
                    if (i == 0)
                        color_enable_r = 4'b0111;
                end
                `TEX_FORMAT_R4G4B4A4: begin
                    formatted_texel_r[i][07:00] = `TEX_COLOR_BITS'(texel_data[i][3:0]);
                    formatted_texel_r[i][23:16] = `TEX_COLOR_BITS'(texel_data[i][7:4]);
                    formatted_texel_r[i][39:32] = `TEX_COLOR_BITS'(texel_data[i][11:8]);
                    formatted_texel_r[i][55:48] = `TEX_COLOR_BITS'(texel_data[i][15:12]);
                    if (i == 0)
                        color_enable_r = 4'b0111;
                end
                `TEX_FORMAT_L8A8: begin
                    formatted_texel_r[i][07:00] = `TEX_COLOR_BITS'(texel_data[i][7:0]);
                    formatted_texel_r[i][23:16] = `TEX_COLOR_BITS'(texel_data[i][15:8]);
                    formatted_texel_r[i][39:32] = `TEX_COLOR_BITS'(0);
                    formatted_texel_r[i][55:48] = `TEX_COLOR_BITS'(0);
                    if (i == 0)
                        color_enable_r = 4'b0011;
                end
                `TEX_FORMAT_A8: begin
                    formatted_texel_r[i][07:00] = `TEX_COLOR_BITS'(texel_data[i][7:0]);
                    formatted_texel_r[i][23:16] = `TEX_COLOR_BITS'(0);
                    formatted_texel_r[i][39:32] = `TEX_COLOR_BITS'(0);
                    formatted_texel_r[i][55:48] = `TEX_COLOR_BITS'(0);
                    if (i == 0)
                        color_enable_r = 4'b0001;
                end
                `TEX_FORMAT_L8: begin
                    formatted_texel_r[i][07:00] = `TEX_COLOR_BITS'(texel_data[i][7:0]);
                    formatted_texel_r[i][23:16] = `TEX_COLOR_BITS'(0);
                    formatted_texel_r[i][39:32] = `TEX_COLOR_BITS'(0);
                    formatted_texel_r[i][55:48] = `TEX_COLOR_BITS'(0);
                    if (i == 0)
                        color_enable_r = 4'b0001;
                end
                default: begin // `TEX_FORMAT_R8G8B8A8:
                    formatted_texel_r[i][07:00] = `TEX_COLOR_BITS'(texel_data[i][7:0]);
                    formatted_texel_r[i][23:16] = `TEX_COLOR_BITS'(texel_data[i][15:8]);
                    formatted_texel_r[i][39:32] = `TEX_COLOR_BITS'(texel_data[i][23:16]);
                    formatted_texel_r[i][55:48] = `TEX_COLOR_BITS'(texel_data[i][31:24]);
                    if (i == 0)
                        color_enable_r = 4'b1111;
                end
            endcase
        end

        // pt sampling direct output
        case (format)
            `TEX_FORMAT_R5G6B5: begin
                formatted_pt_r[07:00] = {`TEX_COLOR_BITS{1'b1}};
                formatted_pt_r[15:08] = `TEX_COLOR_BITS'(texel_data[0][4:0]);
                formatted_pt_r[23:16] = `TEX_COLOR_BITS'(texel_data[0][10:5]);
                formatted_pt_r[31:24] = `TEX_COLOR_BITS'(texel_data[0][15:11]);
            end
            `TEX_FORMAT_R4G4B4A4: begin
                formatted_pt_r[07:00] = `TEX_COLOR_BITS'(texel_data[0][3:0]);
                formatted_pt_r[15:08] = `TEX_COLOR_BITS'(texel_data[0][7:4]);
                formatted_pt_r[23:16] = `TEX_COLOR_BITS'(texel_data[0][11:8]);
                formatted_pt_r[31:24] = `TEX_COLOR_BITS'(texel_data[0][15:12]);
            end
            `TEX_FORMAT_L8A8: begin
                formatted_pt_r[07:00] = `TEX_COLOR_BITS'(texel_data[0][7:0]);
                formatted_pt_r[15:08] = `TEX_COLOR_BITS'(texel_data[0][15:8]);
                formatted_pt_r[23:16] = `TEX_COLOR_BITS'(0);
                formatted_pt_r[31:24] = `TEX_COLOR_BITS'(0);
            end
            `TEX_FORMAT_A8: begin
                formatted_pt_r[07:00] = `TEX_COLOR_BITS'(texel_data[0][7:0]);
                formatted_pt_r[15:08] = `TEX_COLOR_BITS'(0);
                formatted_pt_r[23:16] = `TEX_COLOR_BITS'(0);
                formatted_pt_r[31:24] = `TEX_COLOR_BITS'(0);
            end
            `TEX_FORMAT_L8: begin
                formatted_pt_r[07:00] = `TEX_COLOR_BITS'(texel_data[0][7:0]);
                formatted_pt_r[15:08] = `TEX_COLOR_BITS'(0);
                formatted_pt_r[23:16] = `TEX_COLOR_BITS'(0);
                formatted_pt_r[31:24] = `TEX_COLOR_BITS'(0);
            end
            default: begin // `TEX_FORMAT_R8G8B8A8:
                formatted_pt_r[07:00] = `TEX_COLOR_BITS'(texel_data[0][7:0]);
                formatted_pt_r[15:08] = `TEX_COLOR_BITS'(texel_data[0][15:8]);
                formatted_pt_r[23:16] = `TEX_COLOR_BITS'(texel_data[0][23:16]);
                formatted_pt_r[31:24] = `TEX_COLOR_BITS'(texel_data[0][31:24]);
            end
        endcase



    end

    assign color_enable = color_enable_r;
    assign formatted_pt_texel = formatted_pt_r;
    for (genvar i = 0; i < NUM_TEXELS; i++) begin
        assign formatted_lerp_texel[i] = formatted_texel_r[i] & 64'h00ff00ff00ff00ff;
    end

endmodule
