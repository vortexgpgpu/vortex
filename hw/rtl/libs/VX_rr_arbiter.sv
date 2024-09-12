// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_platform.vh"

`TRACING_OFF
module VX_rr_arbiter #(
    parameter NUM_REQS = 1,
    parameter MODEL    = 1,
    parameter LOG_NUM_REQS = `LOG2UP(NUM_REQS),
    parameter LUT_OPT  = 0
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire [NUM_REQS-1:0]      requests,
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,
    output wire                     grant_valid,
    input  wire                     grant_ready
);
    if (NUM_REQS == 1)  begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (grant_ready)

        assign grant_index  = '0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else if (LUT_OPT && NUM_REQS == 2)  begin

        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;
        reg [LOG_NUM_REQS-1:0]  state;

        always @(*) begin
            casez ({state, requests})
            3'b0_01,
            3'b1_?1: begin grant_onehot_r = 2'b01; grant_index_r = LOG_NUM_REQS'(0); end
            default: begin grant_onehot_r = 2'b10; grant_index_r = LOG_NUM_REQS'(1); end
            endcase
        end

        always @(posedge clk) begin
            if (reset) begin
                state <= '0;
            end else if (grant_ready) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);

    end else if (LUT_OPT && NUM_REQS == 3)  begin

        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;
        reg [LOG_NUM_REQS-1:0]  state;

        always @(*) begin
            casez ({state, requests})
            5'b00_001,
            5'b01_0?1,
            5'b10_??1: begin grant_onehot_r = 3'b001; grant_index_r = LOG_NUM_REQS'(0); end
            5'b00_?1?,
            5'b01_010,
            5'b10_?10: begin grant_onehot_r = 3'b010; grant_index_r = LOG_NUM_REQS'(1); end
            default:   begin grant_onehot_r = 3'b100; grant_index_r = LOG_NUM_REQS'(2); end
            endcase
        end

        always @(posedge clk) begin
            if (reset) begin
                state <= '0;
            end else if (grant_ready) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);

    end else if (LUT_OPT && NUM_REQS == 4)  begin

        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;
        reg [LOG_NUM_REQS-1:0]  state;

        always @(*) begin
            casez ({state, requests})
            6'b00_0001,
            6'b01_00?1,
            6'b10_0??1,
            6'b11_???1: begin grant_onehot_r = 4'b0001; grant_index_r = LOG_NUM_REQS'(0); end
            6'b00_??1?,
            6'b01_0010,
            6'b10_0?10,
            6'b11_??10: begin grant_onehot_r = 4'b0010; grant_index_r = LOG_NUM_REQS'(1); end
            6'b00_?10?,
            6'b01_?1??,
            6'b10_0100,
            6'b11_?100: begin grant_onehot_r = 4'b0100; grant_index_r = LOG_NUM_REQS'(2); end
            default:    begin grant_onehot_r = 4'b1000; grant_index_r = LOG_NUM_REQS'(3); end
            endcase
        end

        always @(posedge clk) begin
            if (reset) begin
                state <= '0;
            end else if (grant_ready) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);

    end else if (LUT_OPT && NUM_REQS == 5)  begin

        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;
        reg [LOG_NUM_REQS-1:0]  state;

        always @(*) begin
            casez ({state, requests})
            8'b000_00001,
            8'b001_000?1,
            8'b010_00??1,
            8'b011_0???1,
            8'b100_????1: begin grant_onehot_r = 5'b00001; grant_index_r = LOG_NUM_REQS'(0); end
            8'b000_???1?,
            8'b001_00010,
            8'b010_00?10,
            8'b011_0??10,
            8'b100_???10: begin grant_onehot_r = 5'b00010; grant_index_r = LOG_NUM_REQS'(1); end
            8'b000_??10?,
            8'b001_??1??,
            8'b010_00100,
            8'b011_0?100,
            8'b100_??100: begin grant_onehot_r = 5'b00100; grant_index_r = LOG_NUM_REQS'(2); end
            8'b000_?100?,
            8'b001_?10??,
            8'b010_?1???,
            8'b011_01000,
            8'b100_?1000: begin grant_onehot_r = 5'b01000; grant_index_r = LOG_NUM_REQS'(3); end
            default:      begin grant_onehot_r = 5'b10000; grant_index_r = LOG_NUM_REQS'(4); end
            endcase
        end

        always @(posedge clk) begin
            if (reset) begin
                state <= '0;
            end else if (grant_ready) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);

    end else if (LUT_OPT && NUM_REQS == 6)  begin

        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;
        reg [LOG_NUM_REQS-1:0]  state;

        always @(*) begin
            casez ({state, requests})
            9'b000_000001,
            9'b001_0000?1,
            9'b010_000??1,
            9'b011_00???1,
            9'b100_0????1,
            9'b101_?????1: begin grant_onehot_r = 6'b000001; grant_index_r = LOG_NUM_REQS'(0); end
            9'b000_????1?,
            9'b001_000010,
            9'b010_000?10,
            9'b011_00??10,
            9'b100_0???10,
            9'b101_????10: begin grant_onehot_r = 6'b000010; grant_index_r = LOG_NUM_REQS'(1); end
            9'b000_???10?,
            9'b001_???1??,
            9'b010_000100,
            9'b011_00?100,
            9'b100_0??100,
            9'b101_???100: begin grant_onehot_r = 6'b000100; grant_index_r = LOG_NUM_REQS'(2); end
            9'b000_??100?,
            9'b001_??10??,
            9'b010_??1???,
            9'b011_001000,
            9'b100_0?1000,
            9'b101_??1000: begin grant_onehot_r = 6'b001000; grant_index_r = LOG_NUM_REQS'(3); end
            9'b000_?1000?,
            9'b001_?100??,
            9'b010_?10???,
            9'b011_?1????,
            9'b100_010000,
            9'b101_?10000: begin grant_onehot_r = 6'b010000; grant_index_r = LOG_NUM_REQS'(4); end
            default:       begin grant_onehot_r = 6'b100000; grant_index_r = LOG_NUM_REQS'(5); end
            endcase
        end

        always @(posedge clk) begin
            if (reset) begin
                state <= '0;
            end else if (grant_ready) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);

    end else if (LUT_OPT && NUM_REQS == 7)  begin

        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;
        reg [LOG_NUM_REQS-1:0]  state;

        always @(*) begin
            casez ({state, requests})
            10'b000_000001,
            10'b001_0000?1,
            10'b010_000??1,
            10'b011_00???1,
            10'b100_00???1,
            10'b101_0????1,
            10'b110_?????1: begin grant_onehot_r = 7'b0000001; grant_index_r = LOG_NUM_REQS'(0); end
            10'b000_?????1?,
            10'b001_0000010,
            10'b010_0000?10,
            10'b011_000??10,
            10'b100_00???10,
            10'b101_0????10,
            10'b110_?????10: begin grant_onehot_r = 7'b0000010; grant_index_r = LOG_NUM_REQS'(1); end
            10'b000_????10?,
            10'b001_????1??,
            10'b010_0000100,
            10'b011_000?100,
            10'b100_00??100,
            10'b101_0???100,
            10'b110_????100: begin grant_onehot_r = 7'b0000100; grant_index_r = LOG_NUM_REQS'(2); end
            10'b000_???100?,
            10'b001_???10??,
            10'b010_???1???,
            10'b011_0001000,
            10'b100_00?1000,
            10'b101_0??1000,
            10'b110_???1000: begin grant_onehot_r = 7'b0001000; grant_index_r = LOG_NUM_REQS'(3); end
            10'b000_??1000?,
            10'b001_??100??,
            10'b010_??10???,
            10'b011_??1????,
            10'b100_0010000,
            10'b101_0?10000,
            10'b110_??10000: begin grant_onehot_r = 7'b0010000; grant_index_r = LOG_NUM_REQS'(4); end
            10'b000_?10000?,
            10'b001_?1000??,
            10'b010_?100???,
            10'b011_?10????,
            10'b100_?1?????,
            10'b101_0100000,
            10'b110_?100000: begin grant_onehot_r = 7'b0100000; grant_index_r = LOG_NUM_REQS'(5); end
            default:         begin grant_onehot_r = 7'b1000000; grant_index_r = LOG_NUM_REQS'(6); end
            endcase
        end

        always @(posedge clk) begin
            if (reset) begin
                state <= '0;
            end else if (grant_ready) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);

    end else if (LUT_OPT && NUM_REQS == 8)  begin

        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;
        reg [LOG_NUM_REQS-1:0]  state;

        always @(*) begin
            casez ({state, requests})
            11'b000_00000001,
            11'b001_000000?1,
            11'b010_00000??1,
            11'b011_0000???1,
            11'b100_000????1,
            11'b101_00?????1,
            11'b110_0??????1,
            11'b111_???????1: begin grant_onehot_r = 8'b00000001; grant_index_r = LOG_NUM_REQS'(0); end
            11'b000_??????1?,
            11'b001_00000010,
            11'b010_00000?10,
            11'b011_0000??10,
            11'b100_000???10,
            11'b101_00????10,
            11'b110_0?????10,
            11'b111_??????10: begin grant_onehot_r = 8'b00000010; grant_index_r = LOG_NUM_REQS'(1); end
            11'b000_?????10?,
            11'b001_?????1??,
            11'b010_00000100,
            11'b011_0000?100,
            11'b100_000??100,
            11'b101_00???100,
            11'b110_0????100,
            11'b111_?????100: begin grant_onehot_r = 8'b00000100; grant_index_r = LOG_NUM_REQS'(2); end
            11'b000_????100?,
            11'b001_????10??,
            11'b010_????1???,
            11'b011_00001000,
            11'b100_000?1000,
            11'b101_00??1000,
            11'b110_0???1000,
            11'b111_????1000: begin grant_onehot_r = 8'b00001000; grant_index_r = LOG_NUM_REQS'(3); end
            11'b000_???1000?,
            11'b001_???100??,
            11'b010_???10???,
            11'b011_???1????,
            11'b100_00010000,
            11'b101_00?10000,
            11'b110_0??10000,
            11'b111_???10000: begin grant_onehot_r = 8'b00010000; grant_index_r = LOG_NUM_REQS'(4); end
            11'b000_??10000?,
            11'b001_??1000??,
            11'b010_??100???,
            11'b011_??10????,
            11'b100_??1?????,
            11'b101_00100000,
            11'b110_0?100000,
            11'b111_??100000: begin grant_onehot_r = 8'b00100000; grant_index_r = LOG_NUM_REQS'(5); end
            11'b000_?100000?,
            11'b001_?10000??,
            11'b010_?1000???,
            11'b011_?100????,
            11'b100_?10?????,
            11'b101_?1??????,
            11'b110_01000000,
            11'b111_?1000000: begin grant_onehot_r = 8'b01000000; grant_index_r = LOG_NUM_REQS'(6); end
            default:          begin grant_onehot_r = 8'b10000000; grant_index_r = LOG_NUM_REQS'(7); end
            endcase
        end

        always @(posedge clk) begin
            if (reset) begin
                state <= '0;
            end else if (grant_ready) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);

    end else if (MODEL == 1) begin

    `IGNORE_UNOPTFLAT_BEGIN
        wire [NUM_REQS-1:0] mask_higher_pri_regs, unmask_higher_pri_regs;
    `IGNORE_UNOPTFLAT_END
        wire [NUM_REQS-1:0] grant_masked, grant_unmasked;

        reg [NUM_REQS-1:0] pointer_reg;

        wire [NUM_REQS-1:0] req_masked = requests & pointer_reg;

        assign mask_higher_pri_regs[0] = 1'b0;
        for (genvar i = 1; i < NUM_REQS; ++i) begin
            assign mask_higher_pri_regs[i] = mask_higher_pri_regs[i-1] | req_masked[i-1];
        end

        assign grant_masked[NUM_REQS-1:0] = req_masked[NUM_REQS-1:0] & ~mask_higher_pri_regs[NUM_REQS-1:0];

        assign unmask_higher_pri_regs[0] = 1'b0;
        for (genvar i = 1; i < NUM_REQS; ++i) begin
            assign unmask_higher_pri_regs[i] = unmask_higher_pri_regs[i-1] | requests[i-1];
        end

        assign grant_unmasked[NUM_REQS-1:0] = requests[NUM_REQS-1:0] & ~unmask_higher_pri_regs[NUM_REQS-1:0];

        wire no_req_masked = ~(|req_masked);
        assign grant_onehot = ({NUM_REQS{no_req_masked}} & grant_unmasked) | grant_masked;

        always @(posedge clk) begin
		    if (reset) begin
				pointer_reg <= {NUM_REQS{1'b1}};
			end else if (grant_ready) begin
				if (|req_masked) begin
                    pointer_reg <= mask_higher_pri_regs;
                end else if (|requests) begin
                    pointer_reg <= unmask_higher_pri_regs;
                end else begin
                    pointer_reg <= pointer_reg;
                end
			end
	    end

        assign grant_valid = (| requests);

        VX_onehot_encoder #(
            .N (NUM_REQS)
        ) onehot_encoder (
            .data_in  (grant_onehot),
            .data_out (grant_index),
            `UNUSED_PIN (valid_out)
        );

    end else begin

        reg [LOG_NUM_REQS-1:0]  grant_index_r;
        reg [NUM_REQS-1:0]      grant_onehot_r;
        reg [NUM_REQS-1:0]      state;

        always @(*) begin
            grant_index_r  = 'x;
            grant_onehot_r = 'x;
            for (integer i = 0; i < NUM_REQS; ++i) begin
                for (integer j = 0; j < NUM_REQS; ++j) begin
                    if (state[i] && requests[(j + 1) % NUM_REQS]) begin
                        grant_index_r  = LOG_NUM_REQS'((j + 1) % NUM_REQS);
                        grant_onehot_r = '0;
                        grant_onehot_r[(j + 1) % NUM_REQS] = 1;
                    end
                end
            end
        end

        always @(posedge clk) begin
            if (reset) begin
                state <= '0;
            end else if (grant_ready) begin
                state <= grant_index_r;
            end
        end

        assign grant_index  = grant_index_r;
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = (| requests);
    end

endmodule
`TRACING_ON
