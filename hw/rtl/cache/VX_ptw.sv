module VX_PTW(
    input wire clk,
    input wire reset,

    input wire tlb_miss, //Needs to be active when tlb_miss happens
    input wire [VPN_WIDTH-1:0] vaddr, //acts on virtual address
    input wire mem_rsp_valid, //to tell that mem access is done
    
    output wire ptw_busy, //outputs ptw busy signal
    output wire [PPN_WIDTH-1:0] paddr, //outputs converted physical address
    output wire [1:0] error_code, //outputs error signal
    output wire ptw_done //outputs ptw done signal
    output wire mem_request //Outputs that memory request has been made
    
    //mem buses
);

enum logic [1:0] {
    IDLE,
    MEM_ACCESS,
    LOOKUP_PT,
    PROPAGATE_ERROR
} state, state_next;

//We will assume 3 level page table (sv39) for now
enum logic [1:0] {
    LEVEL1,
    LEVEL2,
    LEVEL3

    
} pt_level;

// //Internal PTW stages
// enum logic [1:0] {
//     STAGE1,
//     STAGE2_INTERMID,
//     STAGE2_FINAL
// } ptw_stage_q, ptw_stage_n;

enum logic [1:0] {
    NO_ERROR,
    INVALID_PAGE_FAULT,
    NO_LEAF_PAGE_FAULT,
    PERM_PAGE_FAULT     //permission
} error_code_q, error_code_n;


//Start PTW code, peripherals and other things need to bfigured out
always_comb @(*) begin : ptw
    case (state_q)
        IDLE: begin
            //make all things default
            ptw_busy = 1'b0;
            paddr = 64'h0 ; //Can be X ? maybe not
            error_code_n = NO_ERROR;
            done = 1'b0;

            if (tlb_miss) begin
                state_n = MEM_ACCESS;
                //some more things needed to be done here
            end
        end

        MEM_ACCESS: begin
            ptw_busy = 1'b1;
            //when got add response, go to lookup pt
            if (mem_rsp_valid) begin
                state_n = LOOKUP_PT;
            end
        end

        LOOKUP_PT: begin

        end

        PROPAGATE_ERROR: begin
            state_n = IDLE;
            ptw_done = 1'b1;
            ptw_busy = 1'b0; 
        end

        //Do I need to add a default case?
    endcase

end


//mem access needed??
always_ff @(posedge clk) begin 
    if (reset) begin    
        state_q <= IDLE;
        // ptw_stage_q <= STAGE1;
        error_code_q <= NO_ERROR;
    end
    else begin
        state_q <= state_n;
        // ptw_stage_q <= ptw_stage_n;
        error_code_q <= error_code_n;
    end
end

assign error = error_code_q;
assign ptw_done = (state_q == IDLE);
assign ptw_busy = (state_q != IDLE);

endmodule
