#!/usr/bin/env python3

"""Tests for ci/perfetto.py: RTL time base, derived tracks, LMEM level and
process-local async ids. Each test writes a small synthetic RTL log, runs the
exporter's main() on it and inspects the Chrome Trace JSON it produced."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import perfetto  # noqa: E402


def tick(cycle, text, odd=True):
    """An RTL log line at the posedge tick of `cycle` (two ticks per cycle)."""
    return f"{2 * cycle + (1 if odd else 0):>12}: {text}"


class PerfettoRtlTest(unittest.TestCase):

    def export(self, lines, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "run.log"
            dst = pathlib.Path(tmp) / "out.json"
            src.write_text("\n".join(lines) + "\n")
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                rc = perfetto.main([str(src), "-t", "rtlsim", "-o", str(dst), *extra])
            self.assertEqual(rc, 0, err.getvalue())
            self.stderr = err.getvalue()
            return json.loads(dst.read_text())

    @staticmethod
    def names(events):
        return {(e["pid"], e["tid"]): e["args"]["name"] for e in events
                if e.get("ph") == "M" and e.get("name") == "thread_name"}

    def derived(self, events, track_name):
        names = self.names(events)
        return [e for e in events if e.get("pid") == perfetto.DERIVED_PID and e.get("ph") in ("X", "C")
                and names.get((e["pid"], e["tid"])) == track_name]

    # -- time base ------------------------------------------------------------

    def test_rtl_ticks_are_converted_to_cycles(self):
        ev = self.export([
            tick(50, "cluster0-socket0-core0-fetch req: wid=0, PC=0x80000000 (#1)"),
            tick(70, "cluster0-socket0-core0-commit commit: wid=0, PC=0x80000000, ex=ALU, wb=0, sop=1, eop=1 (#1)"),
        ])
        begin = next(e for e in ev if e.get("ph") == "b")
        end = next(e for e in ev if e.get("ph") == "e")
        self.assertEqual((begin["ts"], end["ts"]), (50.0, 70.0))
        commit = next(e for e in ev if e.get("name") == "commit")
        self.assertEqual(commit["args"]["cycle"], 70)

    def test_ticks_per_cycle_is_configurable(self):
        ev = self.export([
            f"{100:>12}: cluster0-socket0-core0-fetch req: wid=0, PC=0x80000000 (#1)",
        ], "--rtl-ticks-per-cycle", "1")
        self.assertEqual(next(e for e in ev if e.get("ph") == "b")["ts"], 100.0)

    def test_async_ids_are_process_local(self):
        ev = self.export([
            tick(1, "cluster0-socket0-core0-fetch req: wid=0, PC=0x80000000 (#7)"),
            tick(2, "cluster0-socket0-core0-commit commit: wid=0, PC=0x80000000, ex=ALU, sop=1, eop=1 (#7)"),
        ])
        for e in ev:
            if e.get("ph") in ("b", "e"):
                self.assertNotIn("id", e)
                self.assertEqual(e["id2"], {"local": "7"})

    # -- memory levels ----------------------------------------------------------

    def test_lmem_events_get_their_own_level(self):
        ev = self.export([tick(3, "cluster0-socket0-core0-lmem bank-rd-req[4]: addr=0xb4, tag=0x9 (#11)")])
        self.assertIn("cluster0-socket0-core0: lmem", self.names(ev).values())
        self.assertEqual(next(e for e in ev if e.get("cat") == "vortex.mem")["args"]["level"], "lmem")

    # -- derived: TCU_OP engine ---------------------------------------------------

    TCU_OP = [
        tick(500, "[tcu_op_core] TCU execution fired, "),
        "A_addr=0xffff0000, B_addr=0xffff2000, C_addr=0xffff7000, D_addr=0x00030000, ",
        tick(510, "LMEM: Issuing read request, req_addr[0]=0xffff7000 mask=1, tag=0 matrix=C, c_blocks_requested=0"),
        tick(790, "LMEM: Issuing read request, req_addr[0]=0xffff0000 mask=1, tag=0 matrix=A, c_blocks_requested=32"),
        *[tick(c, "[tcu_op_core] Issue Busy processing, i_ratio=2, lg_i_ratio=1, ready_to_flush=0")
          for c in (800, 801, 802, 803)],
        tick(800, "c_blk_idx=32, step=0 / 32, set=0, m=0 / 32, n=0 / 32"),
        tick(803, "c_blk_idx=32, step=3 / 32, set=1, m=0 / 32, n=16 / 32"),
        tick(803, "[tcu_op_core] All iterations issued"),
        tick(900, "LMEM: Issuing write request, req_addr[0]=0x00030000 mask=1, tag=0"),
        tick(940, "[tcu_op_core] Result fired downstream"),
        tick(1000, "[tcu_op_core] TCU execution fired, "),
        "A_addr=0xffff8000, B_addr=0xffffa000, C_addr=0xfffff000, D_addr=0x00031000, ",
    ]

    def test_tcu_op_engine_phases(self):
        ev = self.export(self.TCU_OP)
        phases = [(e["name"], e["args"]["cycles"]) for e in self.derived(ev, "TCU_OP: engine phase")]
        self.assertEqual(phases, [("fetch operands", 300), ("issue (compute)", 3),
                                  ("drain + D writeback", 137), ("idle (no MMA_OP)", 60)])
        ops = self.derived(ev, "TCU_OP: MMA_OP lifetime")
        self.assertEqual([(e["name"], e["args"]["cycles"], e["args"]["issue_cycles"]) for e in ops],
                         [("MMA_OP 0", 440, 4)])
        self.assertIn("fetch=300 issue=4 drain=137 idle_between=60", self.stderr)

    def test_issue_busy_counter_marks_gaps(self):
        ev = self.export(self.TCU_OP)
        busy = [(e["ts"], e["args"]["busy"]) for e in self.derived(ev, "TCU_OP: issue busy (1 = FEOP step issued)")]
        self.assertEqual(busy, [(800.0, 1), (804.0, 0)])

    def test_derived_can_be_disabled(self):
        ev = self.export(self.TCU_OP, "--no-derived")
        self.assertFalse([e for e in ev if e.get("pid") == perfetto.DERIVED_PID])

    def test_no_derived_process_without_matching_lines(self):
        ev = self.export([tick(1, "cluster0-socket0-core0-fetch req: wid=0, PC=0x80000000 (#1)")])
        self.assertFalse([e for e in ev if e.get("pid") == perfetto.DERIVED_PID])

    # -- derived: DXA -------------------------------------------------------------

    def test_dxa_service_and_queue_do_not_overlap(self):
        ev = self.export([
            tick(100, "cluster0-socket0-core0-execute-sfu-dxa dxa-req: wid=0, smem=0x7000, meta=0x2, c0=0"),
            tick(102, "cluster0-socket0-dxa dispatch-issue: worker=0, core=0, wid=0, meta=0x2"),
            tick(108, "cluster0-socket0-dxa-worker0 setup-done: core=0, wid=0, bar=0"),
            tick(120, "cluster0-socket0-core0-execute-sfu-dxa dxa-req: wid=0, smem=0x0, meta=0x0, c0=0"),
            tick(122, "cluster0-socket0-dxa dispatch-issue: worker=0, core=0, wid=0, meta=0x0"),
            tick(900, "cluster0-socket0-dxa-worker0 done: core=0, wid=0, bar=0, wr_count=31"),
            tick(901, "cluster0-socket0-dxa-worker0 setup-done: core=0, wid=0, bar=0"),
            tick(1900, "cluster0-socket0-dxa-worker0 done: core=0, wid=0, bar=0, wr_count=63"),
            tick(2000, "[tcu_op_core] TCU execution fired, "),
            "A_addr=0xffff0000, B_addr=0xffff2000, C_addr=0xffff7000, D_addr=0x00030000, ",
        ])
        svc = [(e["name"], e["ts"], e["ts"] + e["dur"]) for e in self.derived(ev, "DXA worker 0: transfer in service")]
        self.assertEqual(svc, [("C buffer 0", 108.0, 900.0), ("A buffer 0", 901.0, 1900.0)])
        queued = [(e["name"], e["args"]["cycles"]) for e in self.derived(ev, "DXA worker 0: transfer queued")]
        self.assertEqual(queued, [("C buffer 0 queued", 6), ("A buffer 0 queued", 779)])

    # -- derived: icache ------------------------------------------------------------

    def test_icache_miss_refill_slice(self):
        ev = self.export([
            tick(200, "cluster0-socket0-icache0-bank0 tags-miss: addr=0x80000440, rw=0, way=0, line=17 (#270)"),
            tick(300, "cluster0-socket0-icache0 core-rd-rsp[0]: tag=0x0, data=0x000c0513 (#270)"),
            tick(310, "cluster0-socket0-icache0 core-rd-rsp[0]: tag=0x0, data=0x00000013 (#271)"),
        ])
        miss = self.derived(ev, "icache miss -> refill (warp fetch blocked)")
        self.assertEqual([(e["name"], e["args"]["cycles"]) for e in miss], [("icache miss 0x80000440", 100)])


if __name__ == "__main__":
    unittest.main()
