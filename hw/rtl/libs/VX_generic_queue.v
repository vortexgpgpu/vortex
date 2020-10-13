`include "VX_platform.vh"

module VX_generic_queue #(
    parameter DATAW    = 1,
    parameter SIZE     = 2,
    parameter BUFFERED = 0,
    parameter ADDRW    = $clog2(SIZE),
    parameter SIZEW    = $clog2(SIZE+1)
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

    always @(*) begin
        assert(!pop || !empty);
        assert(!push || !full);
    end

    if (SIZE == 1) begin // (SIZE == 1)

        reg [SIZEW-1:0] size_r;
        reg [DATAW-1:0] head_r;

        always @(posedge clk) begin
            if (reset) begin
                head_r <= 0;
                size_r <= 0;                    
            end else begin
                if (push && !pop) begin
                    size_r <= 1;
                end else if (pop && !push) begin
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

    end else begin // (SIZE > 1)
    
    `ifdef QUARTUS

        scfifo  scfifo_component (
            .clock (clk),
            .data (data_in),
            .rdreq (pop),
            .wrreq (push),
            .empty (empty),
            .full (full),
            .q (data_out),
            .sclr (reset),
            .usedw (),
            .aclr (),
            .almost_empty (),
            .almost_full (),
            .eccstatus ()            
        );

        defparam
            scfifo_component.lpm_type  = "scfifo",
            scfifo_component.intended_device_family  = "Arria 10",
            scfifo_component.lpm_numwords  = SIZE,            
            scfifo_component.lpm_width  = DATAW,
            scfifo_component.lpm_widthu  = $clog2(SIZE),
            scfifo_component.lpm_showahead  = "ON",            
            scfifo_component.add_ram_output_register = (BUFFERED ? "ON" : "ON"),
            scfifo_component.use_eab  = "ON";

        reg [SIZEW-1:0] size_r;
            
        always @(posedge clk) begin
            if (reset) begin
                size_r   <= 0;
            end else begin
                if (push && !pop) begin                                                       
                    size_r <= size_r + SIZEW'(1);
                end
                if (pop && !push) begin                                                        
                    size_r <= size_r - SIZEW'(1);
                end
            end                   
        end

        assign size = size_r;

    `else

        `USE_FAST_BRAM reg [DATAW-1:0] data [SIZE-1:0];

        if (0 == BUFFERED) begin          

            reg [SIZEW-1:0] size_r;
            reg [ADDRW:0] rd_ptr_r;
            reg [ADDRW:0] wr_ptr_r;
            
            wire [ADDRW-1:0] rd_ptr_a = rd_ptr_r[ADDRW-1:0];
            wire [ADDRW-1:0] wr_ptr_a = wr_ptr_r[ADDRW-1:0];
            
            always @(posedge clk) begin
                if (reset) begin
                    rd_ptr_r <= 0;
                    wr_ptr_r <= 0;
                    size_r   <= 0;
                end else begin
                    if (push) begin            
                        wr_ptr_r <= wr_ptr_r + (ADDRW+1)'(1);
                        if (!pop) begin                                                       
                            size_r <= size_r + SIZEW'(1);
                        end
                    end
                    if (pop) begin
                        rd_ptr_r <= rd_ptr_r + (ADDRW+1)'(1);
                        if (!push) begin                                                        
                            size_r <= size_r - SIZEW'(1);
                        end
                    end
                end                   
            end

            always @(posedge clk) begin
                if (push) begin                             
                    data[wr_ptr_a] <= data_in;
                end                  
            end  

            assign data_out = data[rd_ptr_a];            
            assign empty    = (wr_ptr_r == rd_ptr_r);
            assign full     = (wr_ptr_a == rd_ptr_a) && (wr_ptr_r[ADDRW] != rd_ptr_r[ADDRW]);
            assign size     = size_r;  

        end else begin

            reg [SIZEW-1:0] size_r;
            reg [DATAW-1:0] head_r;
            reg [DATAW-1:0] curr_r;
            reg [ADDRW-1:0] wr_ptr_r;
            reg [ADDRW-1:0] rd_ptr_r;
            reg [ADDRW-1:0] rd_ptr_next_r;
            reg             empty_r;
            reg             full_r;
            reg             bypass_r;

            always @(posedge clk) begin
                if (reset) begin
                    size_r          <= 0;
                    curr_r          <= 0;
                    wr_ptr_r        <= 0;
                    rd_ptr_r        <= 0;
                    rd_ptr_next_r   <= 1;
                    empty_r         <= 1;                   
                    full_r          <= 0;                    
                end else begin
                    if (push) begin                 
                        wr_ptr_r <= wr_ptr_r + ADDRW'(1); 

                        if (!pop) begin                                
                            empty_r <= 0;
                            if (size_r == SIZEW'(SIZE-1)) begin
                                full_r <= 1;
                            end
                            size_r <= size_r + SIZEW'(1);
                        end
                    end

                    if (pop) begin
                        rd_ptr_r <= rd_ptr_next_r;   
                        
                        if (SIZE > 2) begin        
                            rd_ptr_next_r <= rd_ptr_r + ADDRW'(2);
                        end else begin // (SIZE == 2);
                            rd_ptr_next_r <= ~rd_ptr_next_r;                                
                        end

                        if (!push) begin                                
                            if (size_r == SIZEW'(1)) begin
                                assert(rd_ptr_next_r == wr_ptr_r);
                                empty_r <= 1;  
                            end;                
                            full_r <= 0;
                            size_r <= size_r - SIZEW'(1);
                        end
                    end

                    bypass_r <= push && (empty_r || ((size_r == SIZEW'(1)) && pop));                                
                    curr_r   <= data_in;
                end
            end

            always @(posedge clk) begin
                if (reset) begin
                    head_r <= 0;                
                end else begin
                    if (push) begin
                        data[wr_ptr_r] <= data_in;
                    end
                    head_r <= data[pop ? rd_ptr_next_r : rd_ptr_r];
                end
            end 

            assign data_out = bypass_r ? curr_r : head_r;
            assign empty    = empty_r;
            assign full     = full_r;
            assign size     = size_r;
        end

    `endif

    end

endmodule
