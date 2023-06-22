`include "VX_define.vh"

module VX_gbar_unit #(    
    parameter `STRING INSTANCE_ID = ""
) (
    input wire clk,
    input wire reset,

    VX_gbar_bus_if.slave gbar_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam NC_WIDTH = `UP(`NC_BITS);

    reg [`NB_BITS-1:0][`NUM_CORES-1:0] barrier_masks;
    wire [`CLOG2(`NUM_CORES+1)-1:0] active_barrier_count;
    wire [`NUM_CORES-1:0] curr_barrier_mask = barrier_masks[gbar_bus_if.req_id];
    
    `POP_COUNT(active_barrier_count, curr_barrier_mask);
    `UNUSED_VAR (active_barrier_count)

    reg rsp_valid;
    reg [`NB_BITS-1:0] rsp_bar_id;

    always @(posedge clk) begin
        if (reset) begin
            barrier_masks <= '0;
            rsp_valid <= 0;
        end else begin
            if (rsp_valid) begin
                rsp_valid <= 0;
            end
            if (gbar_bus_if.req_valid) begin
                if (active_barrier_count[NC_WIDTH-1:0] == gbar_bus_if.req_size_m1) begin
                    barrier_masks[gbar_bus_if.req_id] <= '0;
                    rsp_bar_id <= gbar_bus_if.req_id;
                    rsp_valid  <= 1;
                end else begin
                    barrier_masks[gbar_bus_if.req_id][gbar_bus_if.req_core_id] <= 1;
                end
            end
        end
    end

    assign gbar_bus_if.rsp_valid = rsp_valid;
    assign gbar_bus_if.rsp_id    = rsp_bar_id;
    assign gbar_bus_if.req_ready = 1; // global barrier unit is always ready (no dependencies)
    
`ifdef DBG_TRACE_GBAR
    always @(posedge clk) begin
        if (gbar_bus_if.req_valid && gbar_bus_if.req_ready) begin
            `TRACE(1, ("%d: %s-acquire: bar_id=%0d, size=%0d, core_id=%0d\n",
                $time, INSTANCE_ID, gbar_bus_if.req_id, gbar_bus_if.req_size_m1, gbar_bus_if.req_core_id));
        end
        if (gbar_bus_if.rsp_valid) begin
            `TRACE(1, ("%d: %s-release: bar_id=%0d\n", $time, INSTANCE_ID, gbar_bus_if.rsp_id));
        end
    end
`endif

endmodule
