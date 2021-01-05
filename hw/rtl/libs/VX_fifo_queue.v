`include "VX_platform.vh"

module VX_fifo_queue #(
    parameter DATAW    = 1,
    parameter SIZE     = 2,
    parameter ADDRW    = $clog2(SIZE),
    parameter SIZEW    = $clog2(SIZE+1),
    parameter BUFFERED = 0,
    parameter FASTRAM  = 1
) ( 
    input  wire             clk,
    input  wire             reset,    
    input  wire             push,
    input  wire             pop,        
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out,
    output wire             empty,
    output wire             full,      
    output wire [SIZEW-1:0] size
); 
    `STATIC_ASSERT(`ISPOW2(SIZE), ("must be 0 or power of 2!"))
    
    if (SIZE == 1) begin

        reg [DATAW-1:0] head_r;
        reg size_r;

        always @(posedge clk) begin
            if (reset) begin
                head_r <= 0;
                size_r <= 0;                    
            end else begin
                if (push && !pop) begin
                    assert(!full);
                    size_r <= 1;
                end else if (pop && !push) begin
                    assert(!empty);
                    size_r <= 0;
                end
                if (push) begin 
                    head_r <= data_in;
                end
            end
        end        

        assign data_out = head_r;
        assign empty    = (size_r == 0);
        assign full     = (size_r != 0);
        assign size     = size_r;

    end else begin
        
        reg empty_r;
        reg full_r;
        reg [ADDRW-1:0] used_r;

        always @(posedge clk) begin
            if (reset) begin
                empty_r <= 1;
                full_r  <= 0;
                used_r  <= 0;
            end else begin
                assert(!push || !full);
                assert(!pop || !empty);
                if (push && !pop) begin
                    empty_r <= 0;
                    if (used_r == ADDRW'(SIZE-1)) begin
                        full_r <= 1;
                    end
                end
                if (pop && !push) begin
                    full_r <= 0;
                    if (used_r == ADDRW'(1)) begin
                        empty_r <= 1;
                    end;
                end
                used_r <= used_r + ADDRW'($signed(2'(push) - 2'(pop)));
            end                   
        end

        if (0 == BUFFERED) begin          

            reg [ADDRW-1:0] rd_ptr_r;
            reg [ADDRW-1:0] wr_ptr_r;
            
            always @(posedge clk) begin
                if (reset) begin
                    rd_ptr_r <= 0;
                    wr_ptr_r <= 0;
                end else begin
                    wr_ptr_r <= wr_ptr_r + ADDRW'(push);
                    rd_ptr_r <= rd_ptr_r + ADDRW'(pop);
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
                .raddr(rd_ptr_r),
                .wren(push),
                .byteen(1'b1),
                .rden(1'b1),
                .din(data_in),
                .dout(data_out)
            );

        end else begin

            wire [DATAW-1:0] dout;
            reg [DATAW-1:0] dout_r;
            reg [ADDRW-1:0] wr_ptr_r;
            reg [ADDRW-1:0] rd_ptr_r;
            reg [ADDRW-1:0] rd_ptr_n_r;

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

            always @(posedge clk) begin
                if (push && (empty_r || ((used_r == ADDRW'(1)) && pop))) begin
                    dout_r <= data_in;
                end else if (pop) begin
                    dout_r <= dout;
                end
            end

            assign data_out = dout_r;
        end
        
        assign empty = empty_r;
        assign full  = full_r;
        assign size  = {full_r, used_r};        
    end

endmodule