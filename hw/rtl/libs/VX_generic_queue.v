`include "VX_define.vh"

module VX_generic_queue #(
    parameter DATAW,
    parameter SIZE = 16,
    parameter BUFFERED_OUTPUT = 1
) ( 
    input  wire             clk,
    input  wire             reset,
    input  wire             push,
    input  wire             pop,        
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out,
    output wire             empty,
    output wire             full,      
    output wire [`LOG2UP(SIZE+1)-1:0] size
); 
    `STATIC_ASSERT(`ISPOW2(SIZE), "must be 0 or power of 2!");

    reg [`LOG2UP(SIZE+1)-1:0] size_r;        
    wire                      reading;
    wire                      writing;

    assign reading = pop && !empty;
    assign writing = push && !full; 

    if (SIZE == 1) begin // (SIZE == 1)

        reg [DATAW-1:0] head_r;

        always @(posedge clk) begin
            if (reset) begin
                head_r <= 0;
                size_r <= 0;                    
            end else begin
                if (writing && !reading) begin
                    size_r <= 1;
                end else if (reading && !writing) begin
                    size_r <= 0;
                end

                if (writing) begin 
                    head_r <= data_in;
                end
            end
        end        

        assign data_out = head_r;
        assign empty    = (size_r == 0);
        assign full     = (size_r != 0);
        assign size     = size_r;

    end else begin // (SIZE > 1)

    `ifdef QUEUE_FORCE_MLAB
        (* syn_ramstyle = "mlab" *) reg [DATAW-1:0] data [SIZE-1:0];
    `else
        reg [DATAW-1:0] data [SIZE-1:0];
    `endif

        if (0 == BUFFERED_OUTPUT) begin                

            reg [`LOG2UP(SIZE):0] wr_ptr_r;
            reg [`LOG2UP(SIZE):0] rd_ptr_r;

            wire [`LOG2UP(SIZE)-1:0] wr_ptr_a = wr_ptr_r[`LOG2UP(SIZE)-1:0];
            wire [`LOG2UP(SIZE)-1:0] rd_ptr_a = rd_ptr_r[`LOG2UP(SIZE)-1:0];

            always @(posedge clk) begin
                if (reset) begin
                    rd_ptr_r <= 0;
                    wr_ptr_r <= 0;
                    size_r   <= 0;
                end else begin
                    if (writing) begin                             
                        data[wr_ptr_a] <= data_in;
                        wr_ptr_r <= wr_ptr_r + 1;

                        if (!reading) begin                                                       
                            size_r <= size_r + 1;
                        end
                    end

                    if (reading) begin
                        rd_ptr_r <= rd_ptr_r + 1;
                        if (!writing) begin                                                        
                            size_r <= size_r - 1;
                        end
                    end
                end                   
            end  

            assign data_out = data[rd_ptr_a];
            assign empty    = (wr_ptr_r == rd_ptr_r);
            assign full     = (wr_ptr_a == rd_ptr_a) && (wr_ptr_r[`LOG2UP(SIZE)] != rd_ptr_r[`LOG2UP(SIZE)]);
            assign size     = size_r;            

        end else begin

            reg [DATAW-1:0]         head_r;
            reg [DATAW-1:0]         curr_r;
            reg [`LOG2UP(SIZE)-1:0] wr_ptr_r;
            reg [`LOG2UP(SIZE)-1:0] rd_ptr_r;
            reg [`LOG2UP(SIZE)-1:0] rd_ptr_next_r;
            reg                     empty_r;
            reg                     full_r;
            reg                     bypass_r;

            always @(posedge clk) begin
                if (reset) begin
                    wr_ptr_r        <= 0;
                    rd_ptr_r        <= 0;
                    rd_ptr_next_r   <= 1;
                    empty_r         <= 1;                   
                    full_r          <= 0;
                    size_r          <= 0;
                end else begin
                    if (writing) begin                            
                        data[wr_ptr_r] <= data_in;
                        wr_ptr_r <= wr_ptr_r + 1; 

                        if (!reading) begin                                
                            empty_r <= 0;
                            if (size_r == ($bits(size_r)'(SIZE-1))) begin
                                full_r <= 1;
                            end
                            size_r <= size_r + 1;
                        end
                    end

                    if (reading) begin
                        rd_ptr_r <= rd_ptr_next_r;   
                        
                        if (SIZE > 2) begin        
                            rd_ptr_next_r <= rd_ptr_r + 2;
                        end else begin // (SIZE == 2);
                            rd_ptr_next_r <= ~rd_ptr_next_r;                                
                        end

                        if (!writing) begin                                
                            if (size_r == 1) begin
                                assert(rd_ptr_next_r == wr_ptr_r);
                                empty_r <= 1;  
                            end;                
                            full_r <= 0;
                            size_r <= size_r - 1;
                        end
                    end

                    bypass_r <= writing 
                                && (empty_r || ((1 == size_r) && reading)); // empty or about to go empty
                                
                    curr_r   <= data_in;
                    head_r   <= data[reading ? rd_ptr_next_r : rd_ptr_r];
                end
            end 

            assign data_out = bypass_r ? curr_r : head_r;
            assign empty    = empty_r;
            assign full     = full_r;
            assign size     = size_r;
        end
    end

endmodule