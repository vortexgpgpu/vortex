/* Leading One Detector for OP TCU */

`include "VX_define.vh"

module VX_tcu_16_way_sorter #(
    parameter UP = 1
) (

    input  wire [15:0] in_bitmap,

    output wire [15:0] out_bitmap,
    output wire [15:0][3:0] out_address,

    output wire [4:0] zero_pop_count  // Total zeros in the 16 elements given
);
    // Split the 16-bit input into two 8-bit groups and reuse the 4-to-8 shifter
    // for each half. The MSB of the address marks the upper 8-bit group.
    wire [7:0]       eight_elem_bitmap  [2];
    wire [7:0][2:0]  eight_elem_address [2];
    wire [3:0]       eight_elem_zeros   [2];

    wire [15:0]      sorted_bitmap;
    wire [15:0][3:0] sorted_address;

    for (genvar i = 0; i < 2; i++) begin : g_shift
        VX_tcu_8_way_sorter #(
            .UP (i == 0 ? 0 : 1) // Lower 8 shifts down, upper 8 shifts up
        ) shifter (
            .in_bitmap      (in_bitmap[8*i +: 8]),
            .out_bitmap     (eight_elem_bitmap[i]),
            .out_address    (eight_elem_address[i]),
            .zero_pop_count (eight_elem_zeros[i])
        );

        assign sorted_bitmap [8*i +: 8] = eight_elem_bitmap[i];
        for (genvar j = 0; j < 8; j++) begin : g_addr
            localparam [0:0] block = i[0];
            assign sorted_address[8*i + j] = eight_elem_bitmap[i][j]
                                            ? {block, eight_elem_address[i][j]}
                                            : 4'd0;
        end
    end

    wire [3:0] zero_pop_count_lsb = eight_elem_zeros[0];
    wire [3:0] zero_pop_count_msb = eight_elem_zeros[1];

    assign zero_pop_count = zero_pop_count_lsb + zero_pop_count_msb;

    generate
        if (UP == 1) begin : g_up
            assign out_bitmap  = sorted_bitmap  >> zero_pop_count_lsb;
            assign out_address = sorted_address >> (4 * zero_pop_count_lsb);
        end else begin : g_down
            assign out_bitmap  = sorted_bitmap  << zero_pop_count_msb;
            assign out_address = sorted_address << (4 * zero_pop_count_msb);
        end
    endgenerate

    // always @(*) begin
    //     `TRACE(1, ("\n%t: [tcu_16_way_sorter] UP=%0d in_bitmap=%b out_bitmap=%b out_addr=0x%x zero_pop_count=%d\n",
    //             $time, UP, in_bitmap, out_bitmap, out_address, zero_pop_count))
    //     `TRACE(1, ("  shifter[0] UP=0 in=%b out=%b addr=%b zeros=%0d\n",
    //             in_bitmap[7:0], eight_elem_bitmap[0], eight_elem_address[0], eight_elem_zeros[0]))
    //     `TRACE(1, ("  shifter[1] UP=1 in=%b out=%b addr=%b zeros=%0d\n",
    //             in_bitmap[15:8], eight_elem_bitmap[1], eight_elem_address[1], eight_elem_zeros[1]))
    // end

endmodule
