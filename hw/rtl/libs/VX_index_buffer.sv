`include "VX_platform.vh"

`TRACING_OFF
module VX_index_buffer #(
    parameter DATAW  = 1,
    parameter SIZE   = 1,
    parameter LUTRAM = 1,
    parameter ADDRW  = `LOG2UP(SIZE)
) (
    input  wire clk,
    input  wire reset,

    output wire [ADDRW-1:0] write_addr,
    input  wire [DATAW-1:0] write_data,            
    input  wire acquire_slot,

    input  wire [ADDRW-1:0] read_addr,
    output wire [DATAW-1:0] read_data,
    input  wire [ADDRW-1:0] release_addr,
    input  wire release_slot,
    
    output wire empty,
    output wire full    
);
    reg [SIZE-1:0] free_slots, free_slots_n;
    reg [ADDRW-1:0] write_addr_r;
    reg empty_r, full_r;
        
    wire free_valid;
    wire [ADDRW-1:0] free_index;

    VX_lzc #(
        .N (SIZE)
    ) free_slots_sel (
        .in_i    (free_slots_n),
        .cnt_o   (free_index),
        .valid_o (free_valid)
    );  

    always @(*) begin
        free_slots_n = free_slots;
        if (release_slot) begin
            free_slots_n[release_addr] = 1;                
        end
        if (acquire_slot) begin
            free_slots_n[write_addr_r] = 0;
        end            
    end    

    always @(posedge clk) begin
        if (reset) begin
            write_addr_r <= ADDRW'(1'b0);
            free_slots   <= {SIZE{1'b1}};
            empty_r      <= 1'b1;
            full_r       <= 1'b0;            
        end else begin
            if (release_slot) begin
                `ASSERT(0 == free_slots[release_addr], ("%t: releasing invalid slot at port %d", $time, release_addr));
            end
            if (acquire_slot) begin
                `ASSERT(1 == free_slots[write_addr], ("%t: acquiring used slot at port %d", $time, write_addr));
            end
            write_addr_r <= free_index;
            free_slots   <= free_slots_n;           
            empty_r      <= (& free_slots_n);
            full_r       <= ~free_valid;
        end        
    end

    VX_dp_ram #(
        .DATAW  (DATAW),
        .SIZE   (SIZE),
        .LUTRAM (LUTRAM)
    ) data_table (
        .clk   (clk), 
        .wren  (acquire_slot),
        .waddr (write_addr_r),
        .wdata (write_data),
        .raddr (read_addr),
        .rdata (read_data)
    );       
        
    assign write_addr = write_addr_r;
    assign empty      = empty_r;
    assign full       = full_r;
    
endmodule
`TRACING_ON
