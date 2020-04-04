`include "VX_define_synth.v"

module VX_generic_queue_ll
	#(
		parameter DATAW = 4,
		parameter SIZE  = 277
	)
	(
	input  wire             clk,
	input  wire             reset,
	input  wire             push,
	input  wire [DATAW-1:0] in_data,

	input  wire             pop,
	output wire [DATAW-1:0] out_data,
	output wire             empty,
	output wire             full
); 

    /* verilator lint_off WIDTH */

    if (SIZE == 0) begin

        assign empty    = 1;
        assign out_data = 0;
        assign full     = 0;

    end else begin // (SIZE > 0)

    `ifdef QUEUE_FORCE_MLAB
        (* syn_ramstyle = "mlab" *) reg[DATAW-1:0] data[SIZE-1:0];
    `else
        reg[ DATAW-1:0] data[SIZE-1:0];
    `endif

        reg [DATAW-1:0]          head_r;        
        reg [$clog2(SIZE+1)-1:0] size_r;        
        wire                     reading;
        wire                     writing;

        assign reading = pop && !empty;
        assign writing = push && !full; 

        if (SIZE == 1) begin

            always @(posedge clk) begin
                if (reset) begin
                    size_r <= 0;
                    head_r <= 0;
                end else begin
                    if (writing && !reading) begin
                        size_r <= 1;
                    end else if (reading && !writing) begin
                        size_r <= 0;
                    end

                    if (writing) begin 
                        head_r <= in_data;
                    end
                end
            end        

            assign out_data = head_r;
            assign empty    = (size_r == 0);
            assign full     = (size_r != 0) && !pop;

        end else begin // (SIZE > 1)

            reg [DATAW-1:0]         curr_r;
            reg [$clog2(SIZE)-1:0]  wr_ctr_r;
            reg [$clog2(SIZE)-1:0]  rd_ptr_r;
            reg [$clog2(SIZE)-1:0]  rd_next_ptr_r;
            reg                     empty_r;
            reg                     full_r;
            reg                     bypass_r;
            
            always @(posedge clk) begin
                if (reset) begin
                    wr_ctr_r <= 0;
                end else begin
                    if (writing)
                    wr_ctr_r <= wr_ctr_r + 1; 
                end 
            end

            always @(posedge clk) begin
                if (reset) begin
                    size_r  <= 0;
                    empty_r <= 1;
                    full_r  <= 0;
                end else begin
                    if (writing && !reading) begin
                        size_r <= size_r + 1;
                        empty_r <= 0;
                        if (size_r == SIZE-1) 
                            full_r <= 1;
                    end else if (reading && !writing) begin
                        size_r <= size_r - 1;
                        if (size_r == 1) 
                            empty_r <= 1;                   
                        full_r <= 0;
                    end
                end
            end    

            always @(posedge clk) begin
                if (writing) begin 
                    data[wr_ctr_r] <= in_data;
                end
            end
       
            always @(posedge clk) begin
                if (reset) begin
                    curr_r        <= 0;
                    rd_ptr_r      <= 0;
                    rd_next_ptr_r <= 1;
                    bypass_r      <= 0;
                end else begin
                    if (reading) begin
                        if (SIZE == 2) begin
                            rd_ptr_r <= rd_next_ptr_r;
                            rd_next_ptr_r <= ~rd_next_ptr_r;
                        end else if (SIZE > 2) begin        
                            rd_ptr_r <= rd_next_ptr_r;
                            rd_next_ptr_r <= rd_ptr_r + 2;
                        end
                    end

                    bypass_r <= writing && (empty_r || (1 == size_r) && reading);
                    curr_r <= in_data;
                    head_r <= data[reading ? rd_next_ptr_r : rd_ptr_r];
                end
            end

            assign out_data = bypass_r ? curr_r : head_r;
            assign empty = empty_r;
            assign full = full_r;
        end
    end

    /* verilator lint_on WIDTH */
    
endmodule