`include "VX_define.vh"

module VX_ibuffer #(
    parameter CORE_ID = 0
) (
    input  wire clk,
    input  wire reset,

    // inputs
    input wire    freeze,       // keep current warp
    VX_decode_if  ibuf_enq_if,  

    // outputs
    VX_decode_if  ibuf_deq_if
);
    localparam DATAW   = `NUM_THREADS + 32 + `EX_BITS + `OP_BITS + `FRM_BITS + 1 + (`NR_BITS * 4) + 32 + 1 + 1 + 1 + `NUM_REGS;
    localparam SIZE    = `IBUF_SIZE;
    localparam SIZEW   = $clog2(SIZE+1);
    localparam ADDRW   = $clog2(SIZE);
    localparam NWARPSW = $clog2(`NUM_WARPS+1);

    `USE_FAST_BRAM reg [DATAW-1:0] entries [`NUM_WARPS-1:0][SIZE-1:0];
    reg [SIZEW-1:0] size_r [`NUM_WARPS-1:0];
    reg [ADDRW:0] rd_ptr_r [`NUM_WARPS-1:0];
    reg [ADDRW:0] wr_ptr_r [`NUM_WARPS-1:0];

    wire [`NUM_WARPS-1:0] q_full;
    wire [`NUM_WARPS-1:0][SIZEW-1:0] q_size;
    wire [DATAW-1:0] q_data_in;
    wire [`NUM_WARPS-1:0][DATAW-1:0] q_data_prev;
    reg [`NUM_WARPS-1:0][DATAW-1:0] q_data_out;

    wire enq_fire = ibuf_enq_if.valid && ibuf_enq_if.ready;
    wire deq_fire = ibuf_deq_if.valid && ibuf_deq_if.ready;

    for (genvar i = 0; i < `NUM_WARPS; ++i) begin

        wire writing = enq_fire && (i == ibuf_enq_if.wid); 
        wire reading = deq_fire && (i == ibuf_deq_if.wid);

        wire [ADDRW-1:0] rd_ptr_a = rd_ptr_r[i][ADDRW-1:0];
        wire [ADDRW-1:0] wr_ptr_a = wr_ptr_r[i][ADDRW-1:0];
        
        always @(posedge clk) begin
            if (reset) begin
                rd_ptr_r[i] <= 0;
                wr_ptr_r[i] <= 0;
                size_r[i]   <= 0;
            end else begin
                if (writing) begin    
                    if ((0 == size_r[i]) || ((1 == size_r[i]) && reading)) begin
                        q_data_out[i] <= q_data_in;
                    end else begin
                        entries[i][wr_ptr_a] <= q_data_in;
                        wr_ptr_r[i] <= wr_ptr_r[i] + ADDRW'(1);
                    end
                    if (!reading) begin                                                       
                        size_r[i] <= size_r[i] + SIZEW'(1);
                    end
                end
                if (reading) begin
                    if (size_r[i] != 1) begin
                        q_data_out[i] <= q_data_prev[i];
                        rd_ptr_r[i]   <= rd_ptr_r[i] + ADDRW'(1);
                    end
                    if (!writing) begin                                                        
                        size_r[i] <= size_r[i] - SIZEW'(1);
                    end
                end
            end                   
        end  

        assign q_data_prev[i] = (wr_ptr_r != rd_ptr_r) ? entries[i][rd_ptr_a] : q_data_in;
        assign q_full[i]      = (size_r[i] == SIZE);
        assign q_size[i]      = size_r[i];
    end

    ///////////////////////////////////////////////////////////////////////////

    reg [`NUM_WARPS-1:0] valid_table, valid_table_n;
    reg [`NUM_WARPS-1:0] schedule_table, schedule_table_n;
    reg [NWARPSW-1:0] num_warps;
    reg [`NW_BITS-1:0] deq_wid, deq_wid_n;
    reg deq_valid, deq_valid_n;
    reg [DATAW-1:0] deq_instr, deq_instr_n;
    
    always @(*) begin
        valid_table_n = valid_table;        
        if (deq_fire) begin
            valid_table_n[ibuf_deq_if.wid] = (q_size[ibuf_deq_if.wid] != 1);
        end
        if (enq_fire) begin
            valid_table_n[ibuf_enq_if.wid] = 1;
        end
    end 

    always @(*) begin         
        deq_wid_n        = 0;
        deq_valid_n      = 0;
        deq_instr_n      = 'x;
        schedule_table_n = schedule_table;       
        if (deq_fire) begin
            schedule_table_n[ibuf_deq_if.wid] = (q_size[ibuf_deq_if.wid] != 1);
        end
        for (integer i = 0; i < `NUM_WARPS; i++) begin
            if (schedule_table_n[i]) begin                
                deq_wid_n   = `NW_BITS'(i);
                deq_valid_n = 1;
                deq_instr_n = (deq_fire && (ibuf_deq_if.wid == `NW_BITS'(i))) ? q_data_prev[i] : q_data_out[i];
                schedule_table_n[i] = 0;
                break;
            end
        end
    end

    wire warp_added   = enq_fire && (0 == q_size[ibuf_enq_if.wid]) && (!deq_fire || ibuf_enq_if.wid != ibuf_deq_if.wid);
    wire warp_removed = deq_fire && (1 == q_size[ibuf_deq_if.wid]) && (!enq_fire || ibuf_enq_if.wid != ibuf_deq_if.wid);
    
    always @(posedge clk) begin
        if (reset)  begin            
            valid_table    <= 0;
            schedule_table <= 0;
            deq_valid      <= 0;  
            num_warps      <= 0;
        end else begin
            valid_table    <= valid_table_n;
            schedule_table <= (| schedule_table_n) ? schedule_table_n : valid_table_n; 

            if (enq_fire && (0 == num_warps)) begin
                deq_valid  <= 1;
                deq_wid    <= ibuf_enq_if.wid;
                deq_instr  <= q_data_in;    
            end else if (!freeze) begin
                deq_valid  <= deq_valid_n;
                deq_wid    <= deq_wid_n;
                deq_instr  <= deq_instr_n; 
            end            

            if (warp_added && !warp_removed) begin
                num_warps <= num_warps + NWARPSW'(1);
            end else if (warp_removed && !warp_added) begin
                num_warps <= num_warps - NWARPSW'(1);                
            end            

        `ifdef VERILATOR
            begin // verify 'num_warps'
                integer nw = 0; 
                for (integer i = 0; i < `NUM_WARPS; i++) begin
                    nw += 32'(q_size[i] != 0);
                end
                assert(nw == 32'(num_warps));
                assert(~deq_fire || num_warps != 0);
            end
        `endif
        end
    end

    assign ibuf_enq_if.ready = ~q_full[ibuf_enq_if.wid];
    assign q_data_in = {ibuf_enq_if.thread_mask, 
                        ibuf_enq_if.curr_PC, 
                        ibuf_enq_if.ex_type, 
                        ibuf_enq_if.ex_op, 
                        ibuf_enq_if.frm, 
                        ibuf_enq_if.wb, 
                        ibuf_enq_if.rd, 
                        ibuf_enq_if.rs1, 
                        ibuf_enq_if.rs2, 
                        ibuf_enq_if.rs3, 
                        ibuf_enq_if.imm, 
                        ibuf_enq_if.rs1_is_PC, 
                        ibuf_enq_if.rs2_is_imm, 
                        ibuf_enq_if.use_rs3, 
                        ibuf_enq_if.used_regs};

    assign ibuf_deq_if.valid = deq_valid;
    assign ibuf_deq_if.wid   = deq_wid;
    assign {ibuf_deq_if.thread_mask, 
            ibuf_deq_if.curr_PC, 
            ibuf_deq_if.ex_type, 
            ibuf_deq_if.ex_op, 
            ibuf_deq_if.frm, 
            ibuf_deq_if.wb, 
            ibuf_deq_if.rd, 
            ibuf_deq_if.rs1, 
            ibuf_deq_if.rs2, 
            ibuf_deq_if.rs3, 
            ibuf_deq_if.imm, 
            ibuf_deq_if.rs1_is_PC, 
            ibuf_deq_if.rs2_is_imm, 
            ibuf_deq_if.use_rs3, 
            ibuf_deq_if.used_regs} = deq_instr;

endmodule