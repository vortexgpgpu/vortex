`include "VX_platform.vh"

module VX_rr_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LOCK_ENABLE  = 0,
    parameter LOG_NUM_REQS = $clog2(NUM_REQS),
    parameter FAST         = 1
) (
    input  wire                     clk,
    input  wire                     reset,          
    input  wire                     enable,
    input  wire [NUM_REQS-1:0]      requests, 
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,   
    output wire                     grant_valid
  );

    if (NUM_REQS == 1)  begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        
        assign grant_index  = 0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else if (FAST == 1)  begin
    
        wire [NUM_REQS-1:0] req_masked;
        wire [NUM_REQS-1:0] grant, grant_masked, grant_unmasked;
    /* verilator lint_off UNOPTFLAT */
        wire [NUM_REQS-1:0] mask_higher_pri_reqs;
    /* verilator lint_off UNOPTFLAT */
        wire [NUM_REQS-1:0] unmask_higher_pri_reqs;
        wire no_req_masked;
        reg [NUM_REQS-1:0] pointer_reg;

        // Simple priority arbitration for masked portion
        assign req_masked = requests & pointer_reg;
        assign mask_higher_pri_reqs[NUM_REQS-1:1] = mask_higher_pri_reqs[NUM_REQS-2:0] | req_masked[NUM_REQS-2:0];
        assign mask_higher_pri_reqs[0] = 1'b0;
        assign grant_masked[NUM_REQS-1:0] = req_masked[NUM_REQS-1:0] & ~mask_higher_pri_reqs[NUM_REQS-1:0];

        // Simple priority arbitration for unmasked portion
        assign unmask_higher_pri_reqs[NUM_REQS-1:1] = unmask_higher_pri_reqs[NUM_REQS-2:0] | requests[NUM_REQS-2:0];
        assign unmask_higher_pri_reqs[0] = 1'b0;
        assign grant_unmasked[NUM_REQS-1:0] = requests[NUM_REQS-1:0] & ~unmask_higher_pri_reqs[NUM_REQS-1:0];
        
        // Use grant_masked if there is any there, otherwise use grant_unmasked.
        assign no_req_masked = ~(| req_masked);
        assign grant = ({NUM_REQS{no_req_masked}} & grant_unmasked) | grant_masked;

        // Generate arbiter pointer update
        wire mask_ptr_sel = (| req_masked) & (!LOCK_ENABLE || enable);
        wire unmask_ptr_sel = (| requests) & (!LOCK_ENABLE || enable);
        
        // Pointer update
        always @(posedge clk) begin
            if (reset) begin
                pointer_reg <= {NUM_REQS{1'b1}};
            end else if (mask_ptr_sel) begin // select if masked arbiter used
                pointer_reg <= mask_higher_pri_reqs;
            end else if (unmask_ptr_sel) begin // select if unmasked arbiter used
                pointer_reg <= unmask_higher_pri_reqs;
            end
        end

        VX_onehot_encoder #(
            .N (NUM_REQS)
        ) onehot_encoder (
            .data_in  (grant),
            .data_out (grant_index),        
            `UNUSED_PIN (valid)
        );

        assign grant_onehot = grant;
        assign grant_valid  = (| requests);

    end else begin

        reg [LOG_NUM_REQS-1:0] grant_table [NUM_REQS-1:0];
        reg [LOG_NUM_REQS-1:0] state;  
        
        always @(*) begin
            for (integer i = 0; i < NUM_REQS; i++) begin  
                grant_table[i] = LOG_NUM_REQS'(i);    
                for (integer j = 0; j < NUM_REQS; j++) begin    
                    if (requests[(i+j) % NUM_REQS]) begin                        
                        grant_table[i] = LOG_NUM_REQS'((i+j) % NUM_REQS);
                    end
                end
            end
        end  

        always @(posedge clk) begin                       
            if (reset) begin         
                state <= 0;
            end else if (!LOCK_ENABLE || enable) begin
                state <= grant_table[state];
            end
        end      

        reg [NUM_REQS-1:0] grant_onehot_r;
        always @(*) begin
            grant_onehot_r = NUM_REQS'(0);
            grant_onehot_r[grant_table[state]] = 1;
        end

        assign grant_index  = grant_table[state];
        assign grant_onehot = grant_onehot_r; 
        assign grant_valid  = (| requests);

    end
    
endmodule