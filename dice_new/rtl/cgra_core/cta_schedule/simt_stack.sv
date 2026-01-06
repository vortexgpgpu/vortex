module simt_stack
  import dice_pkg::*;
  import dice_frontend_pkg::*;
#(
    parameter int STACK_DEPTH = 32,
    parameter int THREAD_WIDTH = DICE_NUM_MAX_THREADS_PER_CORE /
                                 DICE_NUM_MAX_CTA_PER_CORE,
    localparam int StackIndexWidth = $clog2(STACK_DEPTH)
)(
    input logic clk_i,
    input logic rst_i,

    // Push interface (can also modify top when modify_top is asserted)
    input logic push_i,
    input logic modify_top_i,  // When 1, don't increment stack, just update top
    input logic [DICE_ADDR_WIDTH-1:0] push_next_pc_i,
    input logic [DICE_ADDR_WIDTH-1:0] push_reconvergence_pc_i,
    input logic [THREAD_WIDTH-1:0] push_active_mask_i,

    // Pop interface
    input logic pop_i,

    // Read top interface
    input logic read_top_i,  // Request to read top of stack

    // Stack top outputs (registered - valid next cycle after read_top)
    output logic [DICE_ADDR_WIDTH-1:0] top_next_pc_o,
    output logic [DICE_ADDR_WIDTH-1:0] top_reconvergence_pc_o,
    output logic [THREAD_WIDTH-1:0] top_active_mask_o,
    output logic out_valid_o,  // Indicates top outputs are valid

    // Stack status outputs
    output logic stack_empty_o,
    output logic stack_full_o
);

    // Constants
    localparam int EntryWidth = DICE_ADDR_WIDTH + DICE_ADDR_WIDTH +
                                THREAD_WIDTH;

    // Stack pointer (0 = empty, points to top of stack + 1)
    logic [StackIndexWidth:0] stack_ptr_q;  // Extra bit to represent STACK_DEPTH

    // Output valid register
    logic out_valid_q;

    // RAM interface signals
    logic ram_wr_en, ram_rd_en;
    logic [StackIndexWidth-1:0] ram_wr_addr, ram_rd_addr;
    logic [EntryWidth-1:0] ram_wr_data, ram_rd_data;

    // Pack/unpack functions for RAM data
    function automatic [EntryWidth-1:0] pack_entry(
        input logic [DICE_ADDR_WIDTH-1:0] next_pc,
        input logic [DICE_ADDR_WIDTH-1:0] reconvergence_pc,
        input logic [THREAD_WIDTH-1:0] active_mask
    );
        return {next_pc, reconvergence_pc, active_mask};
    endfunction

    function automatic logic [DICE_ADDR_WIDTH-1:0] unpack_next_pc(
        input logic [EntryWidth-1:0] entry);
        return entry[EntryWidth-1:DICE_ADDR_WIDTH+THREAD_WIDTH];
    endfunction

    function automatic logic [DICE_ADDR_WIDTH-1:0] unpack_reconvergence_pc(
        input logic [EntryWidth-1:0] entry);
        return entry[DICE_ADDR_WIDTH+THREAD_WIDTH-1:THREAD_WIDTH];
    endfunction

    function automatic logic [THREAD_WIDTH-1:0] unpack_active_mask(
        input logic [EntryWidth-1:0] entry);
        return entry[THREAD_WIDTH-1:0];
    endfunction

    // Instantiate DICE RAM for stack entries
`ifndef NO_SRAM
    sram_0rw1r1w_320_32_freepdk45 stack_ram (
        .clk0(clk_i),
        .csb0(~ram_wr_en),
        .addr0(ram_wr_addr),
        .din0(ram_wr_data),
        .clk1(clk_i),
        .csb1(~ram_rd_en),
        .addr1(ram_rd_addr),
        .dout1(ram_rd_data)
    );

`else
    dice_ram_1w1r #(
        .DATA_WIDTH(EntryWidth),
        .DEPTH(STACK_DEPTH)
    ) stack_ram (
        .clk(clk_i),
        .wr_en(ram_wr_en),
        .wr_addr(ram_wr_addr),
        .wr_data(ram_wr_data),
        .rd_en(ram_rd_en),
        .rd_addr(ram_rd_addr),
        .rd_data(ram_rd_data)
    );
`endif
    // Stack status
    assign stack_empty_o = (stack_ptr_q == '0);
    assign stack_full_o = (stack_ptr_q == STACK_DEPTH);

    // Top of stack outputs - directly from RAM (registered)
    assign top_next_pc_o = unpack_next_pc(ram_rd_data);
    assign top_reconvergence_pc_o = unpack_reconvergence_pc(ram_rd_data);
    assign top_active_mask_o = unpack_active_mask(ram_rd_data);
    assign out_valid_o = out_valid_q;

    // Control logic for RAM operations
    always_comb begin
        // Default values
        ram_wr_en = 1'b0;
        ram_rd_en = 1'b0;
        ram_wr_addr = '0;
        ram_rd_addr = '0;
        ram_wr_data = '0;

        if ((push_i == 1'b1) && (stack_full_o == 1'b0)) begin
            if ((modify_top_i == 1'b1) && (stack_ptr_q > '0)) begin
                // Modify top: write to current top location
                ram_wr_en = 1'b1;
                ram_wr_addr = (StackIndexWidth)'(stack_ptr_q - 1);
                ram_wr_data = pack_entry(push_next_pc_i, push_reconvergence_pc_i,
                                         push_active_mask_i);
            end else if (modify_top_i == 1'b0) begin
                // Normal push: write to next location
                ram_wr_en = 1'b1;
                ram_wr_addr = (StackIndexWidth)'(stack_ptr_q);
                ram_wr_data = pack_entry(push_next_pc_i, push_reconvergence_pc_i,
                                         push_active_mask_i);
            end
        end

        // Read top of stack when requested
        if ((read_top_i == 1'b1) && (stack_ptr_q > '0)) begin
            ram_rd_en = 1'b1;
            ram_rd_addr = (StackIndexWidth)'(stack_ptr_q - 1);  // Top of stack
        end
    end

    // Sequential logic for stack pointer management and output valid
    always_ff @(posedge clk_i) begin
        if (rst_i == 1'b1) begin
            stack_ptr_q <= '0;
            out_valid_q <= 1'b0;

        end else begin
            // Handle stack pointer updates
            if ((push_i == 1'b1) && (stack_full_o == 1'b0) && (modify_top_i == 1'b0)) begin
                // Normal push: increment stack pointer
                stack_ptr_q <= stack_ptr_q + 1;

            end else if ((pop_i == 1'b1) && (stack_empty_o == 1'b0)) begin
                // Pop: decrement stack pointer
                stack_ptr_q <= stack_ptr_q - 1;
            end
            // modify_top doesn't change stack_ptr_q

            // Handle output valid - becomes valid one cycle after read_top
            if ((read_top_i == 1'b1) && (stack_ptr_q > '0)) begin
                out_valid_q <= 1'b1;
            end else begin
                out_valid_q <= 1'b0;
            end
        end
    end

    // Assertions for debugging
    `ifndef SYNTHESIS
    always @(posedge clk_i) begin
        if (rst_i == 1'b0) begin
            if ((push_i == 1'b1) && (stack_full_o == 1'b1)) begin
                $error("SIMT Stack overflow: trying to push when stack is full");
            end
            if ((pop_i == 1'b1) && (stack_empty_o == 1'b1)) begin
                $error("SIMT Stack underflow: trying to pop empty stack");
            end
            if ((modify_top_i == 1'b1) && (stack_empty_o == 1'b1)) begin
                $error("SIMT Stack: trying to modify top of empty stack");
            end
        end
    end
    `endif

endmodule
