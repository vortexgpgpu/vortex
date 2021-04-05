`include "VX_define.vh"

module VX_ibuffer #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // inputs
    input wire    freeze,       // keep current warp
    VX_decode_if  ibuf_enq_if,  

    // outputs
    VX_decode_if  ibuf_deq_if
);

    `UNUSED_PARAM (CORE_ID)
    
    localparam DATAW   = `NUM_THREADS + 32 + `EX_BITS + `OP_BITS + `FRM_BITS + 1 + (`NR_BITS * 4) + 32 + 1 + 1 + `NUM_REGS;
    localparam SIZE    = 3;
    localparam ADDRW   = $clog2(SIZE);
    localparam NWARPSW = $clog2(`NUM_WARPS+1);

    reg [`NUM_WARPS-1:0][ADDRW-1:0] used_r;
    reg [`NUM_WARPS-1:0] full_r, empty_r, alm_empty_r;
    
    wire [`NUM_WARPS-1:0] q_full, q_empty, q_alm_empty;
    wire [DATAW-1:0] q_data_in;
    wire [`NUM_WARPS-1:0][DATAW-1:0] q_data_prev;    
    reg [`NUM_WARPS-1:0][DATAW-1:0] q_data_out;

    wire enq_fire = ibuf_enq_if.valid && ibuf_enq_if.ready;
    wire deq_fire = ibuf_deq_if.valid && ibuf_deq_if.ready;

    for (genvar i = 0; i < `NUM_WARPS; ++i) begin

        wire writing = enq_fire && (i == ibuf_enq_if.wid); 
        wire reading = deq_fire && (i == ibuf_deq_if.wid);

        wire is_slot0 = empty_r[i] || (alm_empty_r[i] && reading);

        wire push = writing && !is_slot0;
        wire pop = reading && !alm_empty_r[i];

        VX_skid_buffer #(
            .DATAW (DATAW)
        ) queue (
            .clk      (clk),
            .reset    (reset),
            .valid_in (push),
            .data_in  (q_data_in),            
            .ready_out(pop),
            .data_out (q_data_prev[i]),            
            `UNUSED_PIN (ready_in),
            `UNUSED_PIN (valid_out)
        );

        always @(posedge clk) begin
            if (reset) begin            
                used_r[i]      <= 0;
                full_r[i]      <= 0; 
                empty_r[i]     <= 1; 
                alm_empty_r[i] <= 1;
            end else begin  
                if (writing) begin
                    if (!reading) begin
                        empty_r[i] <= 0;
                        if (used_r[i] == 1)
                            alm_empty_r[i] <= 0;
                        if (used_r[i] == ADDRW'(SIZE-1))
                            full_r[i] <= 1;
                    end
                end else if (reading) begin
                    full_r[i] <= 0; 
                    if (used_r[i] == ADDRW'(1))
                        empty_r[i] <= 1;
                    if (used_r[i] == ADDRW'(2))
                        alm_empty_r[i] <= 1;
                end
                used_r[i] <= used_r[i] + ADDRW'($signed(2'(writing) - 2'(reading)));
            end 

            if (writing && is_slot0) begin                                                       
                q_data_out[i] <= q_data_in;
            end else if (pop) begin
                q_data_out[i] <= q_data_prev[i];
            end                  
        end
        
        assign q_full[i]      = full_r[i];
        assign q_empty[i]     = empty_r[i];
        assign q_alm_empty[i] = alm_empty_r[i];
    end

    ///////////////////////////////////////////////////////////////////////////

    reg [`NUM_WARPS-1:0] valid_table, valid_table_n;
    reg [`NUM_WARPS-1:0] schedule_table, schedule_table_n;
    reg [`NW_BITS-1:0] deq_wid, deq_wid_n;
    reg deq_valid, deq_valid_n;
    reg [DATAW-1:0] deq_instr, deq_instr_n;
    reg [NWARPSW-1:0] num_warps;

    always @(*) begin
        valid_table_n = valid_table;        
        if (deq_fire) begin
            valid_table_n[deq_wid] = !q_alm_empty[deq_wid];
        end
        if (enq_fire) begin
            valid_table_n[ibuf_enq_if.wid] = 1;
        end
    end

    // schedule the next instruction to issue
    // do round-robin when multiple warps are active
    always @(*) begin    
        deq_valid_n = 0;     
        deq_wid_n   = 'x;        
        deq_instr_n = 'x;
        schedule_table_n = 'x;         
        
        if ((0 == num_warps) 
         || (1 == num_warps && deq_fire && q_alm_empty[deq_wid])) begin
            deq_valid_n = enq_fire;
            deq_wid_n   = ibuf_enq_if.wid;
            deq_instr_n = q_data_in;  
        end else if ((1 == num_warps) || freeze) begin   
            deq_valid_n = 1;               
            deq_wid_n   = deq_wid;
            deq_instr_n = deq_fire ? q_data_prev[deq_wid] : q_data_out[deq_wid];
        end else begin
            deq_valid_n = (| schedule_table);
            schedule_table_n = schedule_table;
            for (integer i = 0; i < `NUM_WARPS; i++) begin
                if (schedule_table[i]) begin
                    deq_wid_n   = `NW_BITS'(i);                
                    deq_instr_n = q_data_out[i];
                    schedule_table_n[i] = 0;
                    break;
                end
            end
        end   
    end

    wire warp_added   = enq_fire && q_empty[ibuf_enq_if.wid];
    wire warp_removed = deq_fire && ~(enq_fire && ibuf_enq_if.wid == deq_wid) && q_alm_empty[deq_wid];
    
    always @(posedge clk) begin
        if (reset)  begin            
            valid_table    <= 0;
            schedule_table <= 0;
            deq_valid      <= 0;  
            num_warps      <= 0;         
        end else begin
            valid_table <= valid_table_n;

            if (0 == (| schedule_table_n) 
             || (num_warps < 2)) begin
                schedule_table <= valid_table_n;
                schedule_table[deq_wid_n] <= 0;                
            end else begin
                schedule_table <= schedule_table_n;
            end

            deq_valid <= deq_valid_n;

            if (warp_added && !warp_removed) begin
                num_warps <= num_warps + NWARPSW'(1);
            end else if (warp_removed && !warp_added) begin
                num_warps <= num_warps - NWARPSW'(1);                
            end
        end
            
        deq_wid   <= deq_wid_n;
        deq_instr <= deq_instr_n;     
    end
    
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
                        ibuf_enq_if.use_PC, 
                        ibuf_enq_if.use_imm,
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
            ibuf_deq_if.use_PC, 
            ibuf_deq_if.use_imm, 
            ibuf_deq_if.used_regs} = deq_instr;

endmodule