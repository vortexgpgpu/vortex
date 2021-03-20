`include "VX_tex_define.vh"

/*
switch(addressing_mode) {
case undefined:       return is_undefined;
case clamp_to_edge:   return intdowni(max(0, min(coord, coorddim - 1)));
case clamp_to_border: return is_border;
case repeat:
    tile = intdowni(coord / coorddim);
    return intdowni(coord - (tile * coorddim));
case mirrored_repeat:
    mirrored_coord = (coord < 0) ? (-coord - 1) : coord;
    tile = intdowni(mirrored_coord / coorddim);
    mirrored_coord = intdowni(mirrored_coord - (tile * coorddim));
    if (tile & 1) {
        mirrored_coord = (coorddim - 1) - mirrored_coord;
    }
    return mirrored_coord;
}
*/

module VX_tex_wrap #(
    parameter CORE_ID = 0    
) (
    input wire [`TEX_WRAP_BITS-1:0] wrap_i,
    input wire [31:0] coord_i,
    input wire [`FIXED_FRAC-1:0] coord_o
);
    
    `UNUSED_PARAM (CORE_ID)

    reg [31:0] coord_r;

    wire [31:0] clamp = `CLAMP(coord_i, 0, `FIXED_MASK);

    always @(*) begin
        case (wrap_i)
            `TEX_WRAP_CLAMP:   
                coord_r = clamp[`FIXED_FRAC-1:0];
            `TEX_WRAP_MIRROR: 
                coord_r = coord_i[`FIXED_FRAC-1:0] ^ {`FIXED_FRAC{coord_i[`FIXED_FRAC]}};
            default: //`TEX_WRAP_REPEAT
                coord_r = coord_i[`FIXED_FRAC-1:0];
        endcase
    end

    assign coord_o = coord_r;

endmodule