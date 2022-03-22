`include "VX_platform.vh"

`TRACING_OFF
module VX_index_queue #(
    parameter DATAW = 1,
    parameter SIZE = 1
) (
    input  wire clk,
    input  wire reset,
    input  wire [DATAW-1:0] write_data,        
    output wire [`LOG2UP(SIZE)-1:0] write_addr,
    input  wire push,    
    input  wire pop,
    output wire full,
    output wire empty,
    input  wire [`LOG2UP(SIZE)-1:0] read_addr,
    output wire [DATAW-1:0] read_data
);
    reg [DATAW-1:0] entries [SIZE-1:0];
    reg [SIZE-1:0] valid;    
    reg [`LOG2UP(SIZE):0] rd_ptr, wr_ptr;

    wire [`LOG2UP(SIZE)-1:0] rd_a, wr_a;
    wire enqueue, dequeue;

    assign rd_a = rd_ptr[`LOG2UP(SIZE)-1:0];
    assign wr_a = wr_ptr[`LOG2UP(SIZE)-1:0];    

    assign empty = (wr_ptr == rd_ptr);
    assign full  = (wr_a == rd_a) && (wr_ptr[`LOG2UP(SIZE)] != rd_ptr[`LOG2UP(SIZE)]);

    assign enqueue = push;       
    assign dequeue = !empty && !valid[rd_a]; // auto-remove when head is invalid

    `RUNTIME_ASSERT(!push || !full, ("%t: *** invalid inputs", $time));
    
    always @(posedge clk) begin
        if (reset) begin
            rd_ptr <= 0;
            wr_ptr <= 0;
            valid  <= 0;     
        end else begin
            if (enqueue)  begin
                valid[wr_a] <= 1;
                wr_ptr      <= wr_ptr + 1;
            end            
            if (dequeue) begin
                rd_ptr <= rd_ptr + 1;
            end
            if (pop) begin
                valid[read_addr] <= 0;
            end
        end

        if (enqueue)  begin
            entries[wr_a] <= write_data;
        end  
    end

    assign write_addr = wr_a;
    assign read_data = entries[read_addr];

endmodule
`TRACING_ON
