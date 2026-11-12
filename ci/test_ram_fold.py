#!/usr/bin/env python3
"""hw/scripts/ram_fold.py on a small design that crosses every construct the
pass handles, simulated in lockstep against the unfolded design.

The design reads four block-RAM-sized RAMs, one per generate iteration, at an
address that comes from registers in other modules: a library pipe register
with a partial reset, reached through an interface array, a function, a
concatenated assign target and a declaration initializer, plus a hand-written
always_ff register. The four RAMs live in instances of one module, so slang
analyzes one body and the pass maps the other three onto it.

Needs Verilator (VERILATOR or PATH), circt-verilog (CIRCT_VERILOG or PATH) and
pyslang >= 11 in RAM_FOLD_PYTHON (default: this interpreter).
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FOLD = ROOT / "hw/scripts/ram_fold.py"
GEN = ROOT / "hw/scripts/gen_sources.sh"
LIBS = [ROOT / "hw/rtl/VX_platform.vh", ROOT / "hw/rtl/VX_scope.vh",
        ROOT / "hw/rtl/libs/VX_dp_ram.sv", ROOT / "hw/rtl/libs/VX_pipe_register.sv"]

DESIGN = r'''
`include "VX_platform.vh"

typedef struct packed {
    logic [3:0] hi;
    logic [3:0] lo;
} rf_pair_t;

interface rf_if;
    logic [7:0] addr;
    rf_pair_t   pair;
    modport master (output addr, pair);
    modport slave  (input addr, pair);
endinterface

function automatic logic [7:0] rf_swz(input logic [7:0] x);
    return {x[3:0], x[7:4]};
endfunction

module rf_src (
    input wire       clk,
    input wire       reset,
    input wire       en,
    input wire [7:0] din,
    input wire [7:0] din2,
    rf_if            o
);
    VX_pipe_register #(
        .DATAW  (16),
        .RESETW (1)
    ) pr (
        .clk      (clk),
        .reset    (reset),
        .enable   (en),
        .data_in  ({din, din2}),
        .data_out ({o.addr, o.pair})
    );
endmodule

module rf_leaf #(
    parameter SIZE = 256
) (
    input wire        clk,
    input wire        reset,
    input wire        write,
    input wire [7:0]  waddr,
    input wire [15:0] wdata,
    input wire [7:0]  raddr,
    output wire [15:0] rdata
);
    VX_dp_ram #(
        .DATAW    (16),
        .SIZE     (SIZE),
        .RDW_MODE ("R")
    ) ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b0),
        .write (write),
        .wren  (1'b1),
        .waddr (waddr),
        .wdata (wdata),
        .raddr (raddr),
        .rdata (rdata)
    );
endmodule

module rf_leaf_in (
    input wire        clk,
    input wire        reset,
    input wire        write,
    input wire [7:0]  waddr,
    input wire [15:0] wdata,
    input wire [7:0]  raddr,
    output wire [15:0] rdata
);
    VX_dp_ram #(
        .DATAW    (16),
        .SIZE     (256),
        .RDW_MODE ("R")
    ) ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b0),
        .write (write),
        .wren  (1'b1),
        .waddr (waddr),
        .wdata (wdata),
        .raddr (raddr),
        .rdata (rdata)
    );
endmodule

// instantiated once foldable and once not: neither instance may fold
module rf_mixed (
    input wire        clk,
    input wire        reset,
    input wire        write,
    input wire [7:0]  waddr,
    input wire [15:0] wdata,
    input wire [7:0]  raddr,
    output wire [15:0] rdata
);
    VX_dp_ram #(
        .DATAW    (16),
        .SIZE     (256),
        .RDW_MODE ("R")
    ) ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b0),
        .write (write),
        .wren  (1'b1),
        .waddr (waddr),
        .wdata (wdata),
        .raddr (raddr),
        .rdata (rdata)
    );
endmodule

module rf_mid (
    input wire        clk,
    input wire        reset,
    rf_if             i,
    input wire [7:0]  bias,
    input wire        write,
    input wire [7:0]  waddr,
    input wire [15:0] wdata,
    output wire [15:0] rdata
);
    reg [7:0] cnt_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            cnt_r <= '0;
        end else if (write) begin
            cnt_r <= cnt_r + 8'd1;
        end
    end
    wire [3:0] lo_part, hi_part;
    assign {hi_part, lo_part} = i.pair;
    wire [7:0] mixed;
    assign mixed = rf_swz(i.addr) ^ {hi_part, lo_part} ^ cnt_r;
    wire [7:0] addr = mixed + bias;
    rf_leaf leaf (
        .clk   (clk),
        .reset (reset),
        .write (write),
        .waddr (waddr),
        .wdata (wdata),
        .raddr (addr),
        .rdata (rdata)
    );
endmodule

// three addresses the pass must refuse: another clock, a register assigned
// inside a concatenation, a register written by two blocks
module rf_odd (
    input wire        clk,
    input wire        clk2,
    input wire        reset,
    input wire [7:0]  din,
    input wire        write,
    input wire [7:0]  waddr,
    input wire [15:0] wdata,
    output wire [2:0][15:0] rdata
);
    reg [7:0] slow_r, cat_r, tag_r, two_r;
    always_ff @(posedge clk2) begin
        slow_r <= din;
    end
    always_ff @(posedge clk) begin
        {cat_r, tag_r} <= {din, ~din};
    end
    always_ff @(posedge clk) begin
        two_r[3:0] <= din[3:0];
    end
    always_ff @(posedge clk) begin
        two_r[7:4] <= din[7:4];
    end
    VX_dp_ram #(
        .DATAW (16),
        .SIZE  (256)
    ) u_slow (
        .clk   (clk),
        .reset (reset),
        .read  (1'b0),
        .write (write),
        .wren  (1'b1),
        .waddr (waddr),
        .wdata (wdata),
        .raddr (slow_r),
        .rdata (rdata[0])
    );
    VX_dp_ram #(
        .DATAW (16),
        .SIZE  (256)
    ) u_cat (
        .clk   (clk),
        .reset (reset),
        .read  (1'b0),
        .write (write),
        .wren  (1'b1),
        .waddr (waddr),
        .wdata (wdata),
        .raddr (cat_r),
        .rdata (rdata[1])
    );
    VX_dp_ram #(
        .DATAW (16),
        .SIZE  (256)
    ) u_two (
        .clk   (clk),
        .reset (reset),
        .read  (1'b0),
        .write (write),
        .wren  (1'b1),
        .waddr (waddr),
        .wdata (wdata),
        .raddr (two_r),
        .rdata (rdata[2])
    );
endmodule

module rf_top (
    input wire        clk,
    input wire        clk2,
    input wire        reset,
    input wire        en,
    input wire [7:0]  din0,
    input wire [7:0]  din1,
    input wire [7:0]  din2,
    input wire [7:0]  bias_in,
    input wire [7:0]  top_addr,
    input wire        write,
    input wire [7:0]  waddr,
    input wire [15:0] wdata,
    output wire [9:0][15:0] rdata
);
    reg [7:0] bias_r;
    always_ff @(posedge clk) begin
        bias_r <= bias_in;
    end
    rf_if ifs [4] ();
    for (genvar g = 0; g < 4; ++g) begin : g_lane
        rf_src src (
            .clk   (clk),
            .reset (reset),
            .en    (en),
            .din   (din0 + 8'(g)),
            .din2  (g[0] ? din1 : din2),
            .o     (ifs[g])
        );
        rf_mid mid (
            .clk   (clk),
            .reset (reset),
            .i     (ifs[g]),
            .bias  (bias_r),
            .write (write),
            .waddr (waddr),
            .wdata (wdata),
            .rdata (rdata[g])
        );
    end
    rf_mixed mixed_reg (
        .clk   (clk),
        .reset (reset),
        .write (write),
        .waddr (waddr),
        .wdata (wdata),
        .raddr (bias_r),
        .rdata (rdata[5])
    );
    rf_mixed mixed_in (
        .clk   (clk),
        .reset (reset),
        .write (write),
        .waddr (waddr),
        .wdata (wdata),
        .raddr (top_addr),
        .rdata (rdata[6])
    );
    rf_odd odd (
        .clk   (clk),
        .clk2  (clk2),
        .reset (reset),
        .din   (din0),
        .write (write),
        .waddr (waddr),
        .wdata (wdata),
        .rdata (rdata[9:7])
    );
    // read straight from a top-level input: nothing to fold
    rf_leaf_in direct (
        .clk   (clk),
        .reset (reset),
        .write (write),
        .waddr (waddr),
        .wdata (wdata),
        .raddr (top_addr),
        .rdata (rdata[4])
    );
endmodule
'''

TB = r'''
module tb;
    reg clk = 0, clk2 = 0;
    reg reset = 1, en = 1, write = 0;
    reg [7:0] din0 = 0, din1 = 0, din2 = 0, bias_in = 0, top_addr = 0, waddr = 0;
    reg [15:0] wdata = 0;
    wire [9:0][15:0] rdata;
    rf_top dut (.*);
    always #5 clk = ~clk;
    always #7 clk2 = ~clk2;
    integer f, c;
    initial begin
        f = $fopen("trace.txt", "w");
        // every RAM word written before anything is traced
        for (c = 0; c < 256; c = c + 1) begin
            @(negedge clk);
            if (c > 3) reset = 0;
            write = 1; waddr = c[7:0]; wdata = 16'(c * 7919);
        end
        for (c = 0; c < 20000; c = c + 1) begin
            @(negedge clk);
            $fwrite(f, "%0d %h\n", c, rdata);
            en = $urandom_range(3) != 0;
            write = $urandom_range(1);
            din0 = 8'($urandom); din1 = 8'($urandom); din2 = 8'($urandom);
            bias_in = 8'($urandom); top_addr = 8'($urandom);
            waddr = 8'($urandom); wdata = 16'($urandom);
        end
        $fclose(f);
        $finish;
    end
endmodule
'''


def tool(env, name):
    path = os.environ.get(env) or shutil.which(name)
    return path if path and os.path.exists(path) else None


VERILATOR = tool("VERILATOR", "verilator")
CIRCT = tool("CIRCT_VERILOG", "circt-verilog")
PYTHON = os.environ.get("RAM_FOLD_PYTHON", sys.executable)


def have_pyslang():
    return subprocess.run([PYTHON, "-c", "import pyslang.analysis"], capture_output=True).returncode == 0


@unittest.skipUnless(VERILATOR and CIRCT and have_pyslang(),
                     "needs verilator, circt-verilog and pyslang >= 11")
class RamFoldTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="ram_fold_test_"))
        rtl = cls.tmp / "rtl"
        rtl.mkdir()
        for f in LIBS:
            shutil.copy(f, rtl)
        (rtl / "rf_design.sv").write_text(DESIGN)
        (cls.tmp / "tb.sv").write_text(TB)
        env = dict(os.environ, CIRCT_VERILOG=CIRCT, RAM_FOLD_PYTHON=PYTHON)
        # the flow's own path: gen_sources.sh preprocesses, then folds (-F)
        for name, fold in (("orig", []), ("flow", ["-F"])):
            r = subprocess.run(["bash", str(GEN), "-DSIMULATION", "-I" + str(rtl), "-P", *fold,
                                "-Trf_top", "-C" + str(cls.tmp / name)],
                               cwd=cls.tmp, env=env, capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            if fold:
                cls.flow_out = r.stdout
        # the same fold with the next-address check, for simulation
        shutil.copytree(cls.tmp / "orig", cls.tmp / "chk")
        r = subprocess.run([PYTHON, str(FOLD), "--top", "rf_top", "--check", "--diff", "chk.diff",
                            *sorted(str(p) for p in (cls.tmp / "chk").glob("*.sv"))],
                           cwd=cls.tmp, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        cls.report = r.stdout

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def simulate(self, name):
        src = self.tmp / name
        obj = self.tmp / ("obj_" + name)
        files = [str(p) for p in sorted(src.glob("*.sv"))]
        r = subprocess.run([VERILATOR, "--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                            "--x-initial", "0", "--top-module", "tb", "-I" + str(src), "-Mdir", str(obj),
                            *files, str(self.tmp / "tb.sv")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        run = self.tmp / ("run_" + name)
        run.mkdir(exist_ok=True)
        r = subprocess.run([str(obj / "Vtb"), "+verilator+seed+7"], cwd=run, capture_output=True, text=True)
        return r.stdout + r.stderr, (run / "trace.txt").read_text()

    def test_report(self):
        folded = [l for l in self.report.splitlines() if l.startswith("fold ")]
        kept = [l for l in self.report.splitlines() if l.startswith("keep ")]
        self.assertEqual(len(folded), 4, self.report)
        self.assertEqual(len(kept), 6, self.report)
        self.assertTrue(all(".mid.leaf.ram" in l for l in folded), self.report)
        why = {l.split()[1]: l for l in kept}
        self.assertIn("top-level input", why["rf_top.direct.ram"])
        self.assertIn("top-level input", why["rf_top.mixed_in.ram"])
        # the foldable instance shares its instantiation with one that is not
        self.assertIn("rf_top.mixed_in.ram cannot fold", why["rf_top.mixed_reg.ram"])
        self.assertIn("not the RAM's clock", why["rf_top.odd.u_slow"])
        self.assertIn("assigned as part of", why["rf_top.odd.u_cat"])
        self.assertIn("procedural blocks", why["rf_top.odd.u_two"])

    def test_flow_patch_is_the_pass_patch(self):
        # gen_sources.sh -F folds exactly as the pass does on its own
        self.assertEqual(self.flow_out.count("fold "), 4, self.flow_out)
        plain = self.tmp / "plain"
        shutil.copytree(self.tmp / "orig", plain)
        r = subprocess.run([PYTHON, str(FOLD), "--top", "rf_top",
                            *sorted(str(p) for p in plain.glob("*.sv"))], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for f in sorted((self.tmp / "flow").glob("*.sv")):
            self.assertEqual(f.read_text(), (plain / f.name).read_text(), f.name)

    def test_rams_take_the_registered_address_form(self):
        leaf = (self.tmp / "chk" / "rf_design.sv").read_text()
        self.assertIn(".OUT_REG (1)", leaf)
        self.assertIn('.RDW_MODE ("W")', leaf)
        self.assertIn(".raddr (raddr__nx)", leaf)

    def test_lockstep(self):
        out_o, trace_o = self.simulate("orig")
        out_c, trace_c = self.simulate("chk")
        self.assertNotIn("ram_fold:", out_c, out_c[-2000:])
        self.assertEqual(trace_o.count("\n"), 20000)
        self.assertTrue(trace_o == trace_c, "folded design diverges from the original")

    def test_known_bad_patch_is_caught(self):
        # read the counter's current value instead of its next one: a fold off by a cycle
        bad = self.tmp / "bad"
        if not bad.exists():
            shutil.copytree(self.tmp / "chk", bad)
        d = bad / "rf_design.sv"
        text, n = re.subn(r"(assign mixed__nx = .*?)cnt_r__nx", r"\1cnt_r", d.read_text())
        self.assertEqual(n, 1)
        d.write_text(text)
        out_b, trace_b = self.simulate("bad")
        _, trace_o = self.simulate("orig")
        self.assertIn("ram_fold:", out_b)
        self.assertNotEqual(trace_b, trace_o)


if __name__ == "__main__":
    unittest.main()
