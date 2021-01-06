`include "VX_platform.vh"

module VX_input_queue #(
    parameter DATAW    = 1,
    parameter SIZE     = 2,
    parameter ADDRW    = $clog2(SIZE),
    parameter SIZEW    = $clog2(SIZE+1),
    parameter FASTRAM  = 0
) ( 
    input  wire             clk,
    input  wire             reset,    
    input  wire             push,
    input  wire             pop,        
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out,
    output wire             empty,
    output wire [DATAW-1:0] data_out_next,
    output wire             empty_next,
    output wire             full,      
    output wire [SIZEW-1:0] size
); 
    wire [DATAW-1:0] dout;
    reg [DATAW-1:0] dout_r, dout_n_r;
    reg [ADDRW-1:0] wr_ptr_r;
    reg [ADDRW-1:0] rd_ptr_r, rd_ptr_n_r;
    reg full_r;
    reg empty_r, empty_n_r;
    reg [ADDRW-1:0] used_r;  

    always @(*) begin
        empty_n_r = empty_r;
        if (reset) begin
            empty_n_r = 1;
        end else begin
            if (push && !pop) begin
                empty_n_r = 0;
            end
            if (pop && !push) begin
                if (used_r == ADDRW'(1)) begin
                    empty_n_r = 1;
                end;
            end
        end                   
    end 

    always @(*) begin
        dout_n_r = dout_r;
        if (push && (empty_r || ((used_r == ADDRW'(1)) && pop))) begin
            dout_n_r = data_in;
        end else if (pop) begin
            dout_n_r = dout;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            full_r  <= 0;
            used_r  <= 0;
        end else begin
            assert(!push || !full);
            assert(!pop || !empty_r);
            if (push && !pop) begin
                if (used_r == ADDRW'(SIZE-1)) begin
                    full_r <= 1;
                end
            end
            if (pop && !push) begin
                full_r <= 0;
            end
            used_r <= used_r + ADDRW'($signed(2'(push) - 2'(pop)));
        end
        empty_r <= empty_n_r;
        dout_r  <= dout_n_r;                 
    end

    always @(posedge clk) begin
        if (reset) begin      
            wr_ptr_r   <= 0;
            rd_ptr_r   <= 0;
            rd_ptr_n_r <= 1;
        end else begin
            if (push) begin                 
                wr_ptr_r <= wr_ptr_r + ADDRW'(1);
            end
            if (pop) begin
                rd_ptr_r <= rd_ptr_n_r;                           
                if (SIZE > 2) begin        
                    rd_ptr_n_r <= rd_ptr_r + ADDRW'(2);
                end else begin // (SIZE == 2);
                    rd_ptr_n_r <= ~rd_ptr_n_r;                                
                end
            end
        end
    end

    VX_dp_ram #(
        .DATAW(DATAW),
        .SIZE(SIZE),
        .BUFFERED(0),
        .RWCHECK(1),
        .FASTRAM(FASTRAM)
    ) dp_ram (
        .clk(clk),
        .waddr(wr_ptr_r),                                
        .raddr(rd_ptr_n_r),
        .wren(push),
        .byteen(1'b1),
        .rden(1'b1),
        .din(data_in),
        .dout(dout)
    );

    assign data_out      = dout_r;
    assign data_out_next = dout_n_r;    
    assign empty         = empty_r;
    assign empty_next    = empty_n_r;
    assign full          = full_r;
    assign size          = {full_r, used_r};

endmodule