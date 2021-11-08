`include "VX_tex_define.vh"

module VX_tex_stride #(
    parameter CORE_ID = 0
) (
    input wire [`TEX_FORMAT_BITS-1:0]  format,
    output wire [`TEX_STRIDE_BITS-1:0] log_stride
);  
    `UNUSED_PARAM (CORE_ID)

    reg [`TEX_STRIDE_BITS-1:0] log_stride_r;  

    always @(*) begin
        case (format)
            `TEX_FORMAT_A8:       log_stride_r = 0;
            `TEX_FORMAT_L8:       log_stride_r = 0;
            `TEX_FORMAT_L8A8:     log_stride_r = 1;
            `TEX_FORMAT_R5G6B5:   log_stride_r = 1;
            `TEX_FORMAT_R4G4B4A4: log_stride_r = 1;
            //`TEX_FORMAT_R8G8B8A8
            default:              log_stride_r = 2;
        endcase
    end

    assign log_stride = log_stride_r;

endmodule
