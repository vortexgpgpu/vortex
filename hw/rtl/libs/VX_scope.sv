`include "VX_platform.vh"

`TRACING_OFF
module VX_scope #(     
    parameter DATAW  = 64,
    parameter BUSW   = 64,
    parameter SIZE   = 16,
    parameter UPDW   = 1,
    parameter DELTAW = 16
) ( 
    input wire clk,
    input wire reset,
    input wire start,
    input wire stop,
    input wire changed,
    input wire [DATAW-1:0] data_in,
    input wire [BUSW-1:0]  bus_in,
    output wire [BUSW-1:0] bus_out,
    input wire bus_write,
    input wire bus_read
);
    localparam UPDW_ENABLE   = (UPDW != 0);
    localparam MAX_DELTA     = (2 ** DELTAW) - 1;

    localparam CMD_GET_VALID = 3'd0;
    localparam CMD_GET_DATA  = 3'd1;
    localparam CMD_GET_WIDTH = 3'd2;
    localparam CMD_GET_COUNT = 3'd3;
    localparam CMD_SET_START = 3'd4;
    localparam CMD_SET_STOP  = 3'd5;
    localparam CMD_GET_OFFSET= 3'd6;

    localparam GET_VALID  = 3'd0;
    localparam GET_DATA   = 3'd1;
    localparam GET_WIDTH  = 3'd2;
    localparam GET_COUNT  = 3'd3;
    localparam GET_OFFSET = 3'd6;

    `NO_RW_RAM_CHECK reg [DATAW-1:0] data_store [SIZE-1:0];
    `NO_RW_RAM_CHECK reg [DELTAW-1:0] delta_store [SIZE-1:0];

    reg [UPDW-1:0] prev_trigger_id;
    reg [DELTAW-1:0] delta;
    reg [BUSW-1:0]  bus_out_r;
    reg [63:0] timestamp, start_time;

    reg [`CLOG2(SIZE)-1:0] raddr, waddr, waddr_end;

    reg [`LOG2UP(DATAW)-1:0] read_offset;

    reg cmd_start, started, start_wait, recording, data_valid, read_delta, delta_flush;

    reg [BUSW-3:0] delay_val, delay_cntr;

    reg [2:0] get_cmd;
    wire [2:0] cmd_type;
    wire [BUSW-4:0] cmd_data;
    assign {cmd_data, cmd_type} = bus_in;

    wire [UPDW-1:0] trigger_id = data_in[UPDW-1:0];

    always @(posedge clk) begin
        if (reset) begin
            get_cmd         <= $bits(get_cmd)'(CMD_GET_VALID);
            raddr           <= 0;
            waddr           <= 0;
            waddr_end       <= $bits(waddr)'(SIZE-1);
            cmd_start       <= 0;
            started         <= 0;
            start_wait      <= 0;
            recording       <= 0;
            delay_val       <= 0;
            delay_cntr      <= 0;
            delta           <= 0;
            delta_flush     <= 0;
            prev_trigger_id <= 0;
            read_offset     <= 0;
            read_delta      <= 0;
            data_valid      <= 0;
            timestamp       <= 0;
            start_time      <= 0;
        end else begin

            timestamp <= timestamp + 1;

            if (bus_write) begin
                case (cmd_type)
                    CMD_GET_VALID,
                    CMD_GET_DATA,
                    CMD_GET_WIDTH,
                    CMD_GET_OFFSET,
                    CMD_GET_COUNT: get_cmd <= $bits(get_cmd)'(cmd_type);
                    CMD_SET_START: begin 
                        delay_val <= $bits(delay_val)'(cmd_data); 
                        cmd_start <= 1;
                    `ifdef DBG_TRACE_SCOPE
                        dpi_trace("%d: *** scope: CMD_SET_START: delay_val=%0d\n", $time, $bits(delay_val)'(cmd_data));
                    `endif
                    end
                    CMD_SET_STOP: begin
                        waddr_end <= $bits(waddr)'(cmd_data);
                    `ifdef DBG_TRACE_SCOPE
                        dpi_trace("%d: *** scope: CMD_SET_STOP: waddr_end=%0d\n", $time, $bits(waddr)'(cmd_data));
                    `endif
                    end
                    default:;
                endcase
            end

            if (!started && (start || cmd_start)) begin
                started     <= 1;
                delta_flush <= 1;
                if (0 == delay_val) begin
                    start_wait  <= 0;
                    recording   <= 1;
                    delta       <= 0;
                    delay_cntr  <= 0;
                    start_time  <= timestamp;
                `ifdef DBG_TRACE_SCOPE
                    dpi_trace("%d: *** scope: recording start - start_time=%0d\n", $time, timestamp);
                `endif
                end else begin
                    start_wait <= 1;
                    delay_cntr <= delay_val;
                end
            end

            if (start_wait) begin
                delay_cntr <= delay_cntr - 1;
                if (1 == delay_cntr) begin
                    start_wait <= 0;
                    recording  <= 1;
                    delta      <= 0;
                    start_time <= timestamp;
                `ifdef DBG_TRACE_SCOPE
                    dpi_trace("%d: *** scope: recording start - start_time=%0d\n", $time, timestamp);
                `endif
                end 
            end

            if (recording) begin
                if (UPDW_ENABLE) begin
                    if (delta_flush
                     || changed
                     || (trigger_id != prev_trigger_id)) begin
                        delta_store[waddr] <= delta;
                        data_store[waddr]  <= data_in;
                        waddr       <= waddr + $bits(waddr)'(1);
                        delta       <= 0;
                        delta_flush <= 0;
                    end else begin
                        delta       <= delta + DELTAW'(1);
                        delta_flush <= (delta == (MAX_DELTA-1));
                    end
                    prev_trigger_id <= trigger_id;
                end else begin
                    delta_store[waddr] <= 0;
                    data_store[waddr]  <= data_in;
                    waddr <= waddr + 1;
                end

                if (stop
                 || (waddr >= waddr_end)) begin
                `ifdef DBG_TRACE_SCOPE
                    dpi_trace("%d: *** scope: recording stop - waddr=(%0d, %0d)\n", $time, waddr, waddr_end);
                `endif
                    waddr      <= waddr;  // keep last address
                    recording  <= 0;
                    data_valid <= 1;
                    read_delta <= 1;
                end
            end

            if (bus_read
             && (get_cmd == GET_DATA)
             && data_valid)  begin
                if (read_delta) begin
                    read_delta <= 0;
                end else begin
                    if (DATAW > BUSW) begin
                        if (read_offset < $bits(read_offset)'(DATAW-BUSW)) begin
                            read_offset <= read_offset + $bits(read_offset)'(BUSW);
                        end else begin
                            raddr       <= raddr + $bits(raddr)'(1);
                            read_offset <= 0;
                            read_delta  <= 1;
                            if (raddr == waddr) begin
                                data_valid <= 0;
                            end
                        end
                    end else begin
                        raddr <= raddr + 1;
                        read_delta <= 1;
                        if (raddr == waddr) begin
                            data_valid <= 0;
                        end
                    end
                end
            end
        end

        if (recording) begin
            if (UPDW_ENABLE) begin
                if (delta_flush
                 || changed
                 || (trigger_id != prev_trigger_id)) begin
                    delta_store[waddr] <= delta;
                    data_store[waddr]  <= data_in;
                end
            end else begin
                delta_store[waddr] <= 0;
                data_store[waddr]  <= data_in;
            end
        end
    end

    always @(*) begin
        case (get_cmd)
            GET_VALID : bus_out_r = BUSW'(data_valid);
            GET_WIDTH : bus_out_r = BUSW'(DATAW);
            GET_COUNT : bus_out_r = BUSW'(waddr) + BUSW'(1);
            GET_OFFSET: bus_out_r = BUSW'(start_time);
        /* verilator lint_off WIDTH */
            GET_DATA  : bus_out_r = read_delta ? BUSW'(delta_store[raddr]) : BUSW'(data_store[raddr] >> read_offset);
        /* verilator lint_on WIDTH */
            default   : bus_out_r = 0;
        endcase
    end

    assign bus_out = bus_out_r;

`ifdef DBG_TRACE_SCOPE
    always @(posedge clk) begin
        if (bus_read) begin
            dpi_trace("%d: scope-read: cmd=%0d, addr=%0d, value=%0h\n", $time, get_cmd, raddr, bus_out);
        end
        if (bus_write) begin
            dpi_trace("%d: scope-write: cmd=%0d, value=%0d\n", $time, cmd_type, cmd_data);
        end
    end
`endif

endmodule
`TRACING_ON