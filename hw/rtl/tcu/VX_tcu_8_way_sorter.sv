/* Leading One Detector for OP TCU */

`include "VX_define.vh"

module VX_tcu_8_way_sorter #(
    parameter UP = 1
) (

    input  wire [7:0] in_bitmap,

    output wire [7:0] out_bitmap,
    output wire [7:0][2:0] out_address,

    output wire [3:0] zero_pop_count  // Total zeros in the 8 elements given
);
    // Split the 8-bit input into two 4-bit groups and reuse the 4-way sorter
    // for each half. The MSB of the address marks the upper half.
    wire [3:0]      four_elem_bitmap  [2];
    wire [3:0][1:0] four_elem_address [2];

    wire [7:0]      sorted_bitmap;
    wire [7:0][2:0] sorted_address;

    for (genvar i = 0; i < 2; i++) begin : g_sort
        VX_tcu_4_way_sorter #(
            .UP (i == 0 ? 0 : 1) // First 4_way_sorter sorts down, the second sorts up so valids meet in the middle
        ) sorter (
            .in_bitmap   (in_bitmap[4*i +: 4]),
            .out_bitmap  (four_elem_bitmap[i]),
            .out_address (four_elem_address[i])
        );

        assign sorted_bitmap [4*i +: 4]  = four_elem_bitmap[i];
        for (genvar j = 0; j < 4; j++) begin : g_addr
            localparam [0:0] half = i[0];
            assign sorted_address[4*i + j] = four_elem_bitmap[i][j]
                                            ? {half, four_elem_address[i][j]}
                                            : 3'd0;
        end
    end

    wire [2:0] zero_pop_count_lsb;
    wire [2:0] zero_pop_count_msb;
    
    // Count empty slots in the lower 4-way group (to shift valids toward LSB)
    assign zero_pop_count_lsb = (four_elem_bitmap[0] == 4'b0000) ? 3'd4 :
                            (four_elem_bitmap[0] == 4'b1000) ? 3'd3 :
                            (four_elem_bitmap[0] == 4'b1100) ? 3'd2 :
                            (four_elem_bitmap[0] == 4'b1110) ? 3'd1 : 3'd0;
                            
    // Count empty slots in the upper 4-way group (to shift valids toward MSB)
    assign zero_pop_count_msb = (four_elem_bitmap[1] == 4'b0000) ? 3'd4 :
                            (four_elem_bitmap[1] == 4'b0001) ? 3'd3 :
                            (four_elem_bitmap[1] == 4'b0011) ? 3'd2 :
                            (four_elem_bitmap[1] == 4'b0111) ? 3'd1 : 3'd0;

    assign zero_pop_count = zero_pop_count_lsb + zero_pop_count_msb;

    generate
        if (UP == 1) begin : g_up
            assign out_bitmap  = sorted_bitmap  >> zero_pop_count_lsb;
            assign out_address = sorted_address >> (3 * zero_pop_count_lsb);
        end else begin : g_down
            assign out_bitmap  = sorted_bitmap  << zero_pop_count_msb;
            assign out_address = sorted_address << (3 * zero_pop_count_msb);
        end
    endgenerate    


    // always @(*) begin
    //     `TRACE(1, ("\n%t: [tcu_8_way_sorter]: UP=%0d in_bitmap=%b out_bitmap=%b out_addr=0x%x zero_pop_count=%d\n",
    //             $time, UP, in_bitmap, out_bitmap, out_address, zero_pop_count))
    //     `TRACE(1, ("  sorter[0] UP=0 in=%b out=%b addr=%b\n",
    //             in_bitmap[3:0], four_elem_bitmap[0], four_elem_address[0]))
    //     `TRACE(1, ("  sorter[1] UP=1 in=%b out=%b addr=%b\n",
    //             in_bitmap[7:4], four_elem_bitmap[1], four_elem_address[1]))
    // end

endmodule
