`include "VX_tex_define.vh"

module VX_tex_wrap #(
    parameter CORE_ID = 0    
) (
    input wire [`TEX_WRAP_BITS-1:0] wrap_i,
    input wire [31:0] coord_i,
    output wire [`FIXED_FRAC-1:0] coord_o
);
    
    `UNUSED_PARAM (CORE_ID)

    reg [`FIXED_FRAC-1:0] coord_r;

    wire [`FIXED_FRAC-1:0] clamp;

    VX_tex_sat #(
        .IN_W  (32),
        .OUT_W (`FIXED_FRAC)
    ) sat_fx (
        .data_in  (coord_i),
        .data_out (clamp)
    );

    always @(*) begin
        case (wrap_i)
            `TEX_WRAP_CLAMP:   
                coord_r = clamp;
            `TEX_WRAP_MIRROR: 
                coord_r = coord_i[`FIXED_FRAC-1:0] ^ {`FIXED_FRAC{coord_i[`FIXED_FRAC]}};
            default: //`TEX_WRAP_REPEAT
                coord_r = coord_i[`FIXED_FRAC-1:0];
        endcase
    end

    assign coord_o = coord_r;

endmodule