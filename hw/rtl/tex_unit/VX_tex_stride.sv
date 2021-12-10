`include "VX_tex_define.vh"

module VX_tex_stride #(
    parameter CORE_ID = 0
) (
    input wire [`TEX_FORMAT_BITS-1:0]  format,
    output wire [`TEX_LGSTRIDE_BITS-1:0] log_stride
);  
    `UNUSED_PARAM (CORE_ID)

    reg [`TEX_LGSTRIDE_BITS-1:0] log_stride_r;  

    always @(*) begin
        case (format)
            `TEX_FORMAT_A8R8G8B8: log_stride_r = 2;            
            `TEX_FORMAT_R5G6B5,
            `TEX_FORMAT_A1R5G5B5,
            `TEX_FORMAT_A4R4G4B4,
            `TEX_FORMAT_A8L8:     log_stride_r = 1;            
            // `TEX_FORMAT_L8:
            // `TEX_FORMAT_A8:
            default:              log_stride_r = 0;
        endcase
    end

    assign log_stride = log_stride_r;

endmodule
