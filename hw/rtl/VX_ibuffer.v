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
    output wire [`NW_BITS-1:0] deq_wid_next,
    VX_decode_if  ibuf_deq_if
);
    localparam DATAW   = `NUM_THREADS + 32 + `EX_BITS + `OP_BITS + `FRM_BITS + 1 + (`NR_BITS * 4) + 32 + 1 + 1 + `NUM_REGS;
    localparam SIZE    = `IBUF_SIZE;
    localparam SIZEW   = $clog2(SIZE+1);
    localparam ADDRW   = $clog2(SIZE);
    localparam NWARPSW = $clog2(`NUM_WARPS+1);

    reg [`NUM_WARPS-1:0][SIZEW-1:0] size_r;
    
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

        wire is_slot0 = (0 == size_r[i]) || ((1 == size_r[i]) && reading);

        wire push = writing && !is_slot0;
        wire pop = reading && (size_r[i] != 1);

        VX_generic_queue #(
            .DATAW(DATAW),
            .SIZE(SIZE),
            .BUFFERED(1)
        ) queue (
            .clk      (clk),
            .reset    (reset),
            .push     (push),
            .pop      (pop),
            .data_in  (q_data_in),
            .data_out (q_data_prev[i]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (full),
            `UNUSED_PIN (size)
        );

        always @(posedge clk) begin
            if (reset) begin
                size_r[i] <= 0;
            end else begin            
                if (writing && !reading) begin                                                       
                    size_r[i] <= size_r[i] + SIZEW'(1);
                end
                if (reading && !writing) begin                                                        
                    size_r[i] <= size_r[i] - SIZEW'(1);
                end
            end 

            if (writing && is_slot0) begin                                                       
                q_data_out[i] <= q_data_in;
            end
            if (reading && (size_r[i] != 1)) begin
                q_data_out[i] <= q_data_prev[i];
            end                  
        end
        
        assign q_full[i] = (size_r[i] == SIZE);
        assign q_size[i] = size_r[i];
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
            valid_table_n[deq_wid] = (q_size[deq_wid] != SIZEW'(1));
        end
        if (enq_fire) begin
            valid_table_n[ibuf_enq_if.wid] = 1;
        end
    end 

    // schedule the next instruction to issue
    // does round-robin scheduling when multiple warps are present
    always @(*) begin    
        deq_valid_n  = 0;     
        deq_wid_n    = 'x;        
        deq_instr_n  = 'x;

        schedule_table_n = schedule_table;         
        
        if (0 == num_warps) begin
            deq_valid_n  = enq_fire;
            deq_wid_n    = ibuf_enq_if.wid;
            deq_instr_n  = q_data_in;  
        end else if ((1 == num_warps) || freeze) begin   
            deq_valid_n  = (!deq_fire || (q_size[deq_wid] != SIZEW'(1))) || enq_fire;               
            deq_wid_n    = (!deq_fire || (q_size[deq_wid] != SIZEW'(1))) ? deq_wid : ibuf_enq_if.wid;
            deq_instr_n  = deq_fire ? ((q_size[deq_wid] != SIZEW'(1)) ? q_data_prev[deq_wid] : q_data_in) : q_data_out[deq_wid];
        end else begin
            for (integer i = 0; i < `NUM_WARPS; i++) begin
                if (schedule_table_n[i]) begin 
                    deq_valid_n  = 1;               
                    deq_wid_n    = `NW_BITS'(i);                
                    deq_instr_n  = q_data_out[i];
                    schedule_table_n[i] = 0;
                    break;
                end
            end
        end   
    end

    wire warp_added   = enq_fire && (0 == q_size[ibuf_enq_if.wid]);
    wire warp_removed = deq_fire && ~(enq_fire && ibuf_enq_if.wid == deq_wid) && ~(q_size[deq_wid] != SIZEW'(1));
    
    always @(posedge clk) begin
        if (reset)  begin            
            valid_table    <= 0;
            schedule_table <= 0;
            deq_valid      <= 0;  
            num_warps      <= 0;         
        end else begin
            valid_table <= valid_table_n;

            if ((| schedule_table_n)) begin
                schedule_table <= schedule_table_n;
            end else begin
                schedule_table <= valid_table_n;
                schedule_table[deq_wid_n] <= 0;
            end

            deq_valid <= deq_valid_n;
            deq_wid   <= deq_wid_n;
            deq_instr <= deq_instr_n;                 

            if (warp_added && !warp_removed) begin
                num_warps <= num_warps + NWARPSW'(1);
            end else if (warp_removed && !warp_added) begin
                num_warps <= num_warps - NWARPSW'(1);                
            end            

        `ifdef VERILATOR            
            /*if (enq_fire || deq_fire || deq_valid) begin   
                $display("*** %t: cur=%b(%0d), nxt=%b(%0d), enq=%b(%0d), deq=%b(%0d), nw=%0d(%0d,%0d,%0d,%0d), sched=%b, sched_n=%b", 
                $time, deq_valid, deq_wid, deq_valid_n, deq_wid_n, enq_fire, ibuf_enq_if.wid, deq_fire, ibuf_deq_if.wid, num_warps, size_r[0], size_r[1], size_r[2], size_r[3], schedule_table, schedule_table_n);
            end*/
            begin // verify 'num_warps'
                integer nw = 0; 
                for (integer i = 0; i < `NUM_WARPS; i++) begin
                    nw += 32'(q_size[i] != 0);
                end
                assert(nw == 32'(num_warps)) else $error("%t: error: invalid num_warps: nw=%0d, ref=%0d", $time, num_warps, nw);
                assert(~deq_valid || (q_size[deq_wid] != 0)) else $error("%t: error: invalid schedule: wid=%0d", $time, deq_wid);
                assert(~deq_fire || (q_size[deq_wid] != 0)) else $error("%t: error: invalid dequeu: wid=%0d", $time, deq_wid);
            end
        `endif
        end
    end

    assign deq_wid_next = deq_wid_n;

    assign ibuf_enq_if.ready = ~q_full[ibuf_enq_if.wid];
    assign q_data_in = {ibuf_enq_if.tmask, 
                        ibuf_enq_if.PC, 
                        ibuf_enq_if.ex_type, 
                        ibuf_enq_if.op_type, 
                        ibuf_enq_if.op_mod, 
                        ibuf_enq_if.wb, 
                        ibuf_enq_if.rd, 
                        ibuf_enq_if.rs1, 
                        ibuf_enq_if.rs2, 
                        ibuf_enq_if.rs3, 
                        ibuf_enq_if.imm, 
                        ibuf_enq_if.rs1_is_PC, 
                        ibuf_enq_if.rs2_is_imm,
                        ibuf_enq_if.used_regs};

    assign ibuf_deq_if.valid = deq_valid;
    assign ibuf_deq_if.wid   = deq_wid;
    assign {ibuf_deq_if.tmask, 
            ibuf_deq_if.PC, 
            ibuf_deq_if.ex_type, 
            ibuf_deq_if.op_type, 
            ibuf_deq_if.op_mod, 
            ibuf_deq_if.wb, 
            ibuf_deq_if.rd, 
            ibuf_deq_if.rs1, 
            ibuf_deq_if.rs2, 
            ibuf_deq_if.rs3, 
            ibuf_deq_if.imm, 
            ibuf_deq_if.rs1_is_PC, 
            ibuf_deq_if.rs2_is_imm, 
            ibuf_deq_if.used_regs} = deq_instr;

endmodule