`include "VX_platform.vh"

module VX_cam_buffer #(
    parameter DATAW  = 1,
    parameter SIZE   = 1,
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
    
    output wire full
);
    reg [SIZE-1:0] free_slots, free_slots_n;
    reg [ADDRW-1:0] write_addr_r;
    reg full_r;
        
    wire free_valid;
    wire [ADDRW-1:0] free_index;

    VX_priority_encoder #(
        .N(SIZE)
    ) free_slots_encoder (
        .data_in   (free_slots_n),
        .data_out  (free_index),
        .valid_out (free_valid)
    );  

    always @(*) begin
        free_slots_n = free_slots;
        if (release_slot) begin
            free_slots_n[release_addr] = 1;                
        end
        if (acquire_slot)  begin
             assert(1 == free_slots[write_addr]) else $error("%t: acquiring used slot at port %d", $time, write_addr);
             free_slots_n[write_addr_r] = 0;
        end            
    end    

    always @(posedge clk) begin
        if (reset) begin
            free_slots   <= {SIZE{1'b1}};
            full_r       <= 1'b0;
            write_addr_r <= ADDRW'(1'b0);
        end else begin
            if (release_slot) begin
                assert(0 == free_slots[release_addr]) else $error("%t: releasing invalid slot at port %d", $time, release_addr);
            end
            free_slots   <= free_slots_n;
            write_addr_r <= free_index;
            full_r       <= ~free_valid;
        end        
    end

    VX_dp_ram #(
        .DATAW(DATAW),
        .SIZE(SIZE),
        .BUFFERED(0),
        .RWCHECK(0)
    ) data_table (
        .clk(clk),
        .waddr(write_addr),                                
        .raddr(read_addr),
        .wren(acquire_slot),
        .byteen(1'b1),
        .rden(1'b1),
        .din(write_data),
        .dout(read_data)
    );       
        
    assign write_addr = write_addr_r;
    assign full       = full_r;

endmodule