/* Leading One Detector for OP TCU */

`include "VX_define.vh"

module VX_tcu_4_way_sorter #(
    parameter UP = 1
) (
    input  wire [3:0] in_bitmap,

    output wire [3:0] out_bitmap,
    output wire [3:0][1:0] out_address
);
    wire [3:0] next_stage [4];
    wire [2:0] lod_out [4];
    // UP=1 selects from LSB->MSB by reversing the input; UP=0 selects MSB->LSB.
    wire [3:0] lod_in = (UP != 0)
                        ? {in_bitmap[0], in_bitmap[1], in_bitmap[2], in_bitmap[3]}
                        : in_bitmap;

    VX_tcu_lod4 # (
        .ONE_HOT (0)
    ) lod0 (
        .in(lod_in),
        .out(lod_out[0]),
        .next_stage(next_stage[0])
    );
    VX_tcu_lod4 # (
        .ONE_HOT (0)
    ) lod1 (
        .in(next_stage[0]),
        .out(lod_out[1]),
        .next_stage(next_stage[1])
    );
    VX_tcu_lod4  # (
        .ONE_HOT (0)
    ) lod2 (
        .in(next_stage[1]),
        .out(lod_out[2]),
        .next_stage(next_stage[2])
    );
    VX_tcu_lod4  # (
        .ONE_HOT (0)
    ) lod3 (
        .in(next_stage[2]),
        .out(lod_out[3]),
        .next_stage(next_stage[3])
    );

    `UNUSED_VAR (next_stage[3])

    // Decode index (lod_out: 4->bit3, 3->bit2, 2->bit1, 1->bit0, 0->none)
    wire [3:0] out_bitmap_w;
    wire [3:0][1:0] out_address_w;

    for (genvar i = 0; i < 4; i++) begin : g_lod_decode
        localparam int OUT_IDX = (UP != 0) ? i : (3 - i);
        wire has_entry = (lod_out[i] != 3'd0);
        wire [1:0] pos = (lod_out[i] == 3'd4) ? 2'd3 :
                         (lod_out[i] == 3'd3) ? 2'd2 :
                         (lod_out[i] == 3'd2) ? 2'd1 :
                         (lod_out[i] == 3'd1) ? 2'd0 : 2'd0;
        wire [1:0] addr = (UP != 0) ? (2'd3 - pos) : pos;
        assign out_bitmap_w[OUT_IDX]  = has_entry;
        assign out_address_w[OUT_IDX] = has_entry ? addr : 2'd0;
    end

    assign out_bitmap = out_bitmap_w;
    assign out_address = out_address_w;

    // always @(*) begin
    //     `TRACE(1, ("%t: tcu_4_way_sorter in=%b lod={%0d,%0d,%0d,%0d} out_bitmap=%b out_addr={%0d,%0d,%0d,%0d}\n",
    //             $time, in_bitmap,
    //             lod_out[0], lod_out[1], lod_out[2], lod_out[3],
    //             out_bitmap, out_address[3], out_address[2], out_address[1], out_address[0]))
    // end

endmodule
