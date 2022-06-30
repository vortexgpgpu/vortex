`include "VX_platform.vh"

`TRACING_OFF
module VX_fifo_queue_ex #(
    parameter DATAW     = 1,
    parameter SIZE      = 2,
    parameter ALM_FULL  = (SIZE - 1),
    parameter ALM_EMPTY = 1,
    parameter LUTRAM    = 1,
    localparam SIZEW    = $clog2(SIZE+1)
) ( 
    input  wire             clk,
    input  wire             reset,    
    input  wire             push,
    input  wire             pop,        
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out,
    output wire [DATAW-1:0] data_out_n,
    output wire             empty,
    output wire             alm_empty,
    output wire             full,
    output wire             alm_full,
    output wire [SIZEW-1:0] size
); 
    
    localparam ADDRW = $clog2(SIZE);    

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
        assign data_out_n = data_in;
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
                `ASSERT(~(push && ~pop) || ~full, ("runtime error"));
                `ASSERT(~(pop && ~push) || ~empty, ("runtime error"));
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

            reg [DATAW-1:0] data_out_r;
            reg [DATAW-1:0] buffer;
            
            always @(posedge clk) begin
                if (push) begin
                    buffer <= data_in;
                end
                data_out_r <= data_out_n;
            end

            assign data_out_n = (push && (empty_r || (used_r && pop))) ? data_in : (pop ? buffer : data_out_r);
            assign data_out = data_out_r;
        
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
                dout_r <= data_out_n;
            end

            assign data_out_n =  (push && (empty_r || ((used_r == ADDRW'(1)) && pop))) ? data_in : (pop ? dout : dout_r);
            assign data_out = dout_r;
        end
        
        assign empty     = empty_r;        
        assign alm_empty = alm_empty_r;
        assign full      = full_r;
        assign alm_full  = alm_full_r;
        assign size      = {full_r, used_r};        
    end

endmodule
`TRACING_ON
