`include "VX_tex_define.vh"

module VX_tex_wrap (
    input wire [`TEX_WRAP_BITS-1:0]    wrap_i,
    input wire [`VX_TEX_FXD_BITS-1:0]  coord_i,
    output wire [`VX_TEX_FXD_FRAC-1:0] coord_o
);
    
    reg [`VX_TEX_FXD_FRAC-1:0] coord_r;

    wire [`VX_TEX_FXD_FRAC-1:0] clamp;

    VX_tex_sat #(
        .IN_W  (`VX_TEX_FXD_BITS),
        .OUT_W (`VX_TEX_FXD_FRAC)
    ) sat_fx (
        .data_in  (coord_i),
        .data_out (clamp)
    );

    always @(*) begin
        case (wrap_i)
            `VX_TEX_WRAP_CLAMP:   
                coord_r = clamp;
            `VX_TEX_WRAP_MIRROR: 
                coord_r = coord_i[`VX_TEX_FXD_FRAC-1:0] ^ {`VX_TEX_FXD_FRAC{coord_i[`VX_TEX_FXD_FRAC]}};
            default: //`VX_TEX_WRAP_REPEAT
                coord_r = coord_i[`VX_TEX_FXD_FRAC-1:0];
        endcase
    end

    assign coord_o = coord_r;

endmodule
