`include "VX_tex_define.vh"

module VX_tex_format #(
    parameter CORE_ID = 0
) (
    input wire [`TEX_FORMAT_BITS-1:0] format,
    input wire [31:0]   texel_in,    
    output wire [31:0]  texel_out
);  
    `UNUSED_PARAM (CORE_ID)

    reg [31:0] texel_out_r;

    always @(*) begin        
        case (format)
        `TEX_FORMAT_R8G8B8A8: begin
            texel_out_r[07:00] = texel_in[7:0];
            texel_out_r[15:08] = texel_in[15:8];
            texel_out_r[23:16] = texel_in[23:16];
            texel_out_r[31:24] = texel_in[31:24];
        end
        `TEX_FORMAT_R5G6B5: begin
            texel_out_r[07:00] = {texel_in[15:11], texel_in[15:13]};  
            texel_out_r[15:08] = {texel_in[10:5], texel_in[10:9]};
            texel_out_r[23:16] = {texel_in[4:0], texel_in[4:2]};
            texel_out_r[31:24] = 8'hff;
        end
        `TEX_FORMAT_R4G4B4A4: begin
            texel_out_r[07:00] = {texel_in[11:8], texel_in[15:12]};
            texel_out_r[15:08] = {2{texel_in[7:4]}};
            texel_out_r[23:16] = {2{texel_in[3:0]}};
            texel_out_r[31:24] = {2{texel_in[15:12]}};
        end
        `TEX_FORMAT_L8A8: begin
            texel_out_r[07:00] = texel_in[7:0];
            texel_out_r[15:08] = texel_in[7:0];
            texel_out_r[23:16] = texel_in[7:0];
            texel_out_r[31:24] = texel_in[15:8];
        end
        `TEX_FORMAT_L8: begin
            texel_out_r[07:00] = texel_in[7:0];
            texel_out_r[15:08] = texel_in[7:0];
            texel_out_r[23:16] = texel_in[7:0];
            texel_out_r[31:24] = 8'hff;
        end
        //`TEX_FORMAT_A8
        default: begin
            texel_out_r[07:00] = 0;
            texel_out_r[15:08] = 0;
            texel_out_r[23:16] = 0;
            texel_out_r[31:24] = texel_in[7:0];
        end
        endcase
    end

    assign texel_out = texel_out_r;

endmodule
