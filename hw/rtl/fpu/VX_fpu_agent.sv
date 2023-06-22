`include "VX_define.vh"
`include "VX_fpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_fpu_types::*;
`IGNORE_WARNINGS_END

module VX_fpu_agent #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    VX_fpu_agent_if.slave   fpu_agent_if,
    VX_fpu_to_csr_if.master fpu_to_csr_if,
    VX_commit_if.master     fpu_commit_if,

    VX_fpu_bus_if.master    fpu_bus_if,

    input wire              csr_pending,
    output wire             req_pending
);
    `UNUSED_PARAM (CORE_ID)
    
    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam NW_WIDTH   = `UP(`NW_BITS);
    
    // Store request info

    wire [UUID_WIDTH-1:0]   rsp_uuid;
    wire [NW_WIDTH-1:0]     rsp_wid;
    wire [`NUM_THREADS-1:0] rsp_tmask;
    wire [`XLEN-1:0]        rsp_PC;
    wire [`NR_BITS-1:0]     rsp_rd;

    wire [`FPU_REQ_TAG_WIDTH-1:0] req_tag, rsp_tag;    
    wire mdata_full;

    wire fpu_agent_fire = fpu_agent_if.valid && fpu_agent_if.ready;
    wire fpu_rsp_fire = fpu_bus_if.rsp_valid && fpu_bus_if.rsp_ready;

    assign rsp_tag = fpu_bus_if.rsp_tag;

    VX_index_buffer #(
        .DATAW   (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS),
        .SIZE    (`FPU_REQ_QUEUE_SIZE)
    ) tag_store  (
        .clk          (clk),
        .reset        (reset),
        .acquire_slot (fpu_agent_fire),       
        .write_addr   (req_tag),                
        .read_addr    (rsp_tag),
        .release_addr (rsp_tag),        
        .write_data   ({fpu_agent_if.uuid, fpu_agent_if.wid, fpu_agent_if.tmask, fpu_agent_if.PC, fpu_agent_if.rd}),                    
        .read_data    ({rsp_uuid,          rsp_wid,          rsp_tmask,          rsp_PC,          rsp_rd}), 
        .release_slot (fpu_rsp_fire),     
        .full         (mdata_full),
        `UNUSED_PIN (empty)
    );

    // resolve dynamic FRM from CSR   
    wire [`INST_FRM_BITS-1:0] req_op_frm;
    assign fpu_to_csr_if.read_wid = fpu_agent_if.wid;    
    assign req_op_frm = (fpu_agent_if.op_type != `INST_FPU_MISC 
                      && fpu_agent_if.frm == `INST_FRM_DYN) ? fpu_to_csr_if.read_frm : fpu_agent_if.frm;

    // submit FPU request

    wire mdata_and_csr_ready = ~mdata_full && ~csr_pending;

    wire valid_in, ready_in;    
    assign valid_in = fpu_agent_if.valid && mdata_and_csr_ready;
    assign fpu_agent_if.ready = ready_in && mdata_and_csr_ready;    

    VX_skid_buffer #(
        .DATAW   (`INST_FPU_BITS + `INST_FMT_BITS + `INST_FRM_BITS + `NUM_THREADS * 3 * `XLEN + `FPU_REQ_TAG_WIDTH),
        .OUT_REG (1)
    ) req_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_in),
        .ready_in  (ready_in),
        .data_in   ({fpu_agent_if.op_type, fpu_agent_if.fmt,   req_op_frm,         fpu_agent_if.rs1_data, fpu_agent_if.rs2_data, fpu_agent_if.rs3_data, req_tag}),
        .data_out  ({fpu_bus_if.req_type,  fpu_bus_if.req_fmt, fpu_bus_if.req_frm, fpu_bus_if.req_dataa,  fpu_bus_if.req_datab,  fpu_bus_if.req_datac,  fpu_bus_if.req_tag}),
        .valid_out (fpu_bus_if.req_valid),
        .ready_out (fpu_bus_if.req_ready)
    );

    // handle FPU response

    fflags_t rsp_fflags;
    always @(*) begin
        rsp_fflags = '0;        
        for (integer i = 0; i < `NUM_THREADS; ++i) begin
            if (rsp_tmask[i]) begin
                rsp_fflags.NX |= fpu_bus_if.rsp_fflags[i].NX;
                rsp_fflags.UF |= fpu_bus_if.rsp_fflags[i].UF;
                rsp_fflags.OF |= fpu_bus_if.rsp_fflags[i].OF;
                rsp_fflags.DZ |= fpu_bus_if.rsp_fflags[i].DZ;
                rsp_fflags.NV |= fpu_bus_if.rsp_fflags[i].NV;
            end
        end
    end
    
    assign fpu_to_csr_if.write_enable = fpu_rsp_fire && fpu_bus_if.rsp_has_fflags;
    assign fpu_to_csr_if.write_wid    = rsp_wid;     
    assign fpu_to_csr_if.write_fflags = rsp_fflags;

    // commit

    VX_skid_buffer #(
        .DATAW (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS + (`NUM_THREADS * `XLEN))
    ) rsp_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fpu_bus_if.rsp_valid),
        .ready_in  (fpu_bus_if.rsp_ready),
        .data_in   ({rsp_uuid,           rsp_wid,           rsp_tmask,           rsp_PC,           rsp_rd,           fpu_bus_if.rsp_result}),
        .data_out  ({fpu_commit_if.uuid, fpu_commit_if.wid, fpu_commit_if.tmask, fpu_commit_if.PC, fpu_commit_if.rd, fpu_commit_if.data}),
        .valid_out (fpu_commit_if.valid),
        .ready_out (fpu_commit_if.ready)
    );

    assign fpu_commit_if.wb  = 1'b1; 
    assign fpu_commit_if.eop = 1'b1;

    // pending request

    wire fpu_commit_fire = fpu_commit_if.valid && fpu_commit_if.ready;

    reg req_pending_r;
    always @(posedge clk) begin
        if (reset) begin
            req_pending_r <= 0;
        end else begin                      
            if (fpu_agent_fire) begin
                 req_pending_r <= 1;
            end
            if (fpu_commit_fire) begin
                 req_pending_r <= 0;
            end
        end
    end
    assign req_pending = req_pending_r;

endmodule
