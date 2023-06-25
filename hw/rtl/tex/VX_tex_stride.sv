`include "VX_tex_define.vh"

module VX_tex_stride (
    input wire [`TEX_FORMAT_BITS-1:0]    format,
    output wire [`TEX_LGSTRIDE_BITS-1:0] log_stride
);  

    reg [`TEX_LGSTRIDE_BITS-1:0] log_stride_r;  

    always @(*) begin
        case (format)
            `VX_TEX_FORMAT_A8R8G8B8: log_stride_r = 2;            
            `VX_TEX_FORMAT_R5G6B5,
            `VX_TEX_FORMAT_A1R5G5B5,
            `VX_TEX_FORMAT_A4R4G4B4,
            `VX_TEX_FORMAT_A8L8:     log_stride_r = 1;            
            `VX_TEX_FORMAT_L8,
            `VX_TEX_FORMAT_A8:       log_stride_r = 0;
            default:                 log_stride_r = 'x;
        endcase
    end

    assign log_stride = log_stride_r;

endmodule
