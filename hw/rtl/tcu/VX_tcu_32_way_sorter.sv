/* Leading One Detector for OP TCU */

`include "VX_define.vh"

module VX_tcu_32_way_sorter #(
    parameter UP = 1
) (

    input  wire [31:0] in_bitmap,

    output wire [31:0] out_bitmap,
    output wire [31:0][4:0] out_address,

    output wire [5:0] zero_pop_count  // Total zeros in the 32 elements given
);
    // Split the 32-bit input into two 16-bit groups and reuse the 8-to-16 shifter
    // for each half. The MSB of the address marks the upper 16-bit group.
    wire [15:0]       sixteen_elem_bitmap  [2];
    wire [15:0][3:0]  sixteen_elem_address [2];
    wire [4:0]        sixteen_elem_zeros   [2];

    wire [31:0]      sorted_bitmap;
    wire [31:0][4:0] sorted_address;

    for (genvar i = 0; i < 2; i++) begin : g_shift
        VX_tcu_16_way_sorter #(
            .UP (i == 0 ? 0 : 1) // Lower 16 shifts down, upper 16 shifts up
        ) shifter (
            .in_bitmap      (in_bitmap[16*i +: 16]),
            .out_bitmap     (sixteen_elem_bitmap[i]),
            .out_address    (sixteen_elem_address[i]),
            .zero_pop_count (sixteen_elem_zeros[i])
        );

        assign sorted_bitmap [16*i +: 16] = sixteen_elem_bitmap[i];
        for (genvar j = 0; j < 16; j++) begin : g_addr
            localparam [0:0] block = i[0];
            assign sorted_address[16*i + j] = sixteen_elem_bitmap[i][j]
                                             ? {block, sixteen_elem_address[i][j]}
                                             : 5'd0;
        end
    end

    wire [4:0] zero_pop_count_lsb = sixteen_elem_zeros[0];
    wire [4:0] zero_pop_count_msb = sixteen_elem_zeros[1];

    assign zero_pop_count = zero_pop_count_lsb + zero_pop_count_msb;

    generate
        if (UP == 1) begin : g_up
            assign out_bitmap  = sorted_bitmap  >> zero_pop_count_lsb;
            assign out_address = sorted_address >> (5 * zero_pop_count_lsb);
        end else begin : g_down
            assign out_bitmap  = sorted_bitmap  << zero_pop_count_msb;
            assign out_address = sorted_address << (5 * zero_pop_count_msb);
        end
    endgenerate

    // always @(*) begin
    //     `TRACE(1, ("%t: [tcu_32_way_sorter]: UP=%0d in_bitmap=%b out_bitmap=%b out_addr=0x%x zero_pop_count=%d\n",
    //             $time, UP, in_bitmap, out_bitmap, out_address, zero_pop_count))
    //     `TRACE(1, ("  shifter[0] UP=0 in=%b out=%b addr=%b zeros=%0d\n",
    //             in_bitmap[15:0], sixteen_elem_bitmap[0], sixteen_elem_address[0], sixteen_elem_zeros[0]))
    //     `TRACE(1, ("  shifter[1] UP=1 in=%b out=%b addr=%b zeros=%0d\n",
    //             in_bitmap[31:16], sixteen_elem_bitmap[1], sixteen_elem_address[1], sixteen_elem_zeros[1]))
    // end

endmodule
