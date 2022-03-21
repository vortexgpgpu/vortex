`include "VX_platform.vh"

`TRACING_OFF
module VX_fifo_queue #(
    parameter DATAW     = 1,
    parameter SIZE      = 2,
    parameter ALM_FULL  = (SIZE - 1),
    parameter ALM_EMPTY = 1,
    parameter ADDRW     = $clog2(SIZE),
    parameter SIZEW     = $clog2(SIZE+1),
    parameter OUT_REG   = 0,
    parameter LUTRAM    = 1
) ( 
    input  wire             clk,
    input  wire             reset,    
    input  wire             push,
    input  wire             pop,        
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out,
    output wire             empty,      
    output wire             alm_empty,
    output wire             full,            
    output wire             alm_full,
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
                `ASSERT(!push || !full, ("runtime error"));
                `ASSERT(!pop || !empty, ("runtime error"));
                if (push) begin
                    if (!pop) begin
                        size_r <= 1;
                    end
                end else if (pop) begin
                    size_r <= 0;
                end
                if (push) begin 
                    head_r <= data_in;
                end
            end
        end        

        assign data_out  = head_r;
        assign empty     = (size_r == 0);
        assign alm_empty = 1'b1;
        assign full      = (size_r != 0);
        assign alm_full  = 1'b1;
        assign size      = size_r;

    end else begin
        
        reg empty_r, alm_empty_r;
        reg full_r, alm_full_r;
        reg [ADDRW-1:0] used_r;

        always @(posedge clk) begin
            if (reset) begin
                empty_r     <= 1;
                alm_empty_r <= 1;    
                full_r      <= 0;        
                alm_full_r  <= 0;
                used_r      <= 0;
            end else begin
                `ASSERT(!push || !full, ("runtime error"));
                `ASSERT(!pop || !empty, ("runtime error"));
                if (push) begin
                    if (!pop) begin
                        empty_r <= 0;
                        if (used_r == ADDRW'(ALM_EMPTY))
                            alm_empty_r <= 0;
                        if (used_r == ADDRW'(SIZE-1))
                            full_r <= 1;
                        if (used_r == ADDRW'(ALM_FULL-1))
                            alm_full_r <= 1;
                    end
                end else if (pop) begin
                    full_r <= 0;        
                    if (used_r == ADDRW'(ALM_FULL))
                        alm_full_r <= 0;            
                    if (used_r == ADDRW'(1))
                        empty_r <= 1;
                    if (used_r == ADDRW'(ALM_EMPTY+1))
                        alm_empty_r <= 1;
                end
                if (SIZE > 2) begin        
                    used_r <= used_r + ADDRW'($signed(2'(push) - 2'(pop)));
                end else begin 
                    // (SIZE == 2);
                    used_r[0] <= used_r[0] ^ (push ^ pop);
                end                
            end                   
        end

        if (SIZE == 2) begin

            if (0 == OUT_REG) begin 

                reg [DATAW-1:0] shift_reg [1:0];

                always @(posedge clk) begin
                    if (push) begin
                        shift_reg[1] <= shift_reg[0];
                        shift_reg[0] <= data_in;
                    end
                end

                assign data_out = shift_reg[!used_r[0]];    
                
            end else begin

                reg [DATAW-1:0] data_out_r;
                reg [DATAW-1:0] buffer;

                always @(posedge clk) begin
                    if (push) begin
                        buffer <= data_in;
                    end
                    if (push && (empty_r || (used_r && pop))) begin
                        data_out_r <= data_in;
                    end else if (pop) begin
                        data_out_r <= buffer;
                    end
                end

                assign data_out = data_out_r;

            end
        
        end else begin

            if (0 == OUT_REG) begin          

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
                    .DATAW   (DATAW),
                    .SIZE    (SIZE),
                    .OUT_REG (0),
                    .LUTRAM  (LUTRAM)
                ) dp_ram (
                    .clk(clk),
                    .wren  (push),
                    .waddr (wr_ptr_r),
                    .wdata (data_in),
                    .raddr (rd_ptr_r),
                    .rdata (data_out)
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
                    .DATAW   (DATAW),
                    .SIZE    (SIZE),
                    .OUT_REG (0),
                    .LUTRAM  (LUTRAM)
                ) dp_ram (
                    .clk   (clk),
                    .wren  (push),
                    .waddr (wr_ptr_r),
                    .wdata (data_in),
                    .raddr (rd_ptr_n_r),
                    .rdata (dout)
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
        end
        
        assign empty     = empty_r;        
        assign alm_empty = alm_empty_r;
        assign full      = full_r;
        assign alm_full  = alm_full_r;
        assign size      = {full_r, used_r};        
    end

endmodule
`TRACING_ON
