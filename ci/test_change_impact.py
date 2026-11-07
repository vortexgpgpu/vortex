#!/usr/bin/env python3
# Copyright © 2019-2026
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""ci/change_impact.py on synthetic diffs, plus a replay over real history.

  python3 -m pytest ci/test_change_impact.py

Each synthetic case commits a base tree and a head tree to a scratch git
repository and asserts the class, the contract findings and the evidence. The
replay cases pin classifications of real commits, so a detector change that
moves them is a reviewed diff; they skip when the commits are not in the clone.
"""

import json
import os
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import change_impact as ci  # noqa: E402

POLICY = ci.load_policy(ci.DEFAULT_POLICY)

IFACE = """`include "VX_define.vh"

interface VX_demo_if import VX_gpu_pkg::*; #(
    parameter type data_t = logic
) ();

    logic  valid;
    data_t data;
    logic  ready;

    modport master (
        output valid,
        output data,
        input  ready
    );

    modport slave (
        input  valid,
        input  data,
        output ready
    );

endinterface
"""

PKG = """package VX_gpu_pkg;
    localparam NUM_LANES = 4;
    localparam INST_ALU_ADD = 4'b0000;
    localparam INST_ALU_SUB = 4'b0001;

    typedef struct packed {
        logic [3:0] op;
        logic [31:0] addr;
    } req_t;

endpackage
"""

UNIT = """module VX_unit (
    input wire clk,
    input wire reset
);
    reg [3:0] count_r;
    always @(posedge clk) begin
        if (reset) begin
            count_r <= '0;
        end else begin
            count_r <= count_r + 1;
        end
    end
endmodule
"""

SIMX = """#include "unit.h"

void Unit::tick() {
  if (queue_.empty()) {
    return;
  }
  delay_ = 2;
}
"""

HEADER = """#ifndef VX_DEMO_H
#define VX_DEMO_H
#ifdef __cplusplus
extern "C" {
#endif

#define VX_DEMO_FLAG 1

typedef void* vx_device_h;

int vx_dev_open(vx_device_h* hdevice);
int vx_dev_close(vx_device_h hdevice);
int vx_printf(const char * format, ...);

#ifdef __cplusplus
}
#endif
#endif
"""

ABI = """#pragma once
#include <stdint.h>

typedef struct {
  uint32_t x;
  uint32_t y;
} rast_prim_t;
"""

TYPES = """[globals]
VX_CSR_ADDR_BITS = 12

[csr_base]
VX_CSR_FOO = 0x7C0
"""

TESTCASES = """category: demo
defaults:
  tier: smoke
tests:
  - id: parity-sgemm
    app: sgemm
    check: model_parity
    args: "-n64 -m32"
    tolerance: 0.03

  - id: vecadd
    app: vecadd
    drivers: [simx, rtlsim]
"""

DEFINE = """`ifndef VX_DEFINE_VH
`define VX_DEFINE_VH
`define INST_FENCE_D 1'b0
`define NUM_REGS 32
`endif
"""

DECODE_SV = """module VX_decode;
    always @(*) begin
        case (funct3)
            3'h0: r_type = INST_ALU_ADD;
            3'h1: r_type = INST_ALU_SUB;
            default: r_type = 'x;
        endcase
    end
endmodule
"""

DECODE_CPP = """void Emulator::decode(uint32_t code) {
  switch (op) {
  case Opcode::R:
    instr->set_op_type(funct3 ? AluType::SUB : AluType::ADD);
    break;
  default:
    break;
  }
}
"""

INTRINSICS = """#pragma once
#include <stdint.h>

inline void vx_fence() {
  __asm__ volatile (".insn r 0x0b, 0, 0, x0, x0, x0" ::: "memory");
}
"""

CONFIG = """[base]
VX_CFG_NUM_WARPS = 4
VX_CFG_NUM_THREADS = 4
VX_CFG_ISSUE_WIDTH = "expr: up($VX_CFG_NUM_WARPS / 16)"
"""

CONFIGURE = """#!/bin/bash
# defaults
default_xlen=32
CONFIGS="-DVX_CFG_XLEN=$default_xlen"
echo done
"""

GEN_CONFIG = """import sys


def resolve(value):
    return value
"""

MAKEFILE_IN = """KERNEL_INC_DST := $(PREFIX)/kernel/include
SW_COMMON_PUBLIC := vx_gfx_abi.h gfx_dcr.h

install: $(KERNEL_INC_DST)
\tcp -r sw/kernel/include $(KERNEL_INC_DST)

all:
\t$(MAKE) -C sw
"""

KERNEL_MK = """export TOOLDIR ?= $(HOME)/tools
CFLAGS += -march=rv32imaf -mabi=ilp32f
CFLAGS += -O3 -Wall

all: libvortex2.a
"""

WORKFLOW = """name: Demo
on:
  pull_request:
  workflow_dispatch:
jobs:
  build:
    runs-on: ubuntu-22.04
    steps:
      - run: echo build
"""

WORKFLOW_PUSH_FIRST = """name: Later
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'
  pull_request:
jobs:
  build:
    runs-on: ubuntu-22.04
    steps:
      - run: |
          echo build
"""

GIT_ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")


class Scratch:
    def __init__(self, path):
        self.path = str(path)
        self._git("init", "-q", "-b", "main")
        self._git("config", "user.email", "ci@example.invalid")
        self._git("config", "user.name", "ci")

    def _git(self, *args):
        return subprocess.run(["git", "-C", self.path] + list(args), check=True, env=GIT_ENV,
                              stdout=subprocess.PIPE, text=True).stdout.strip()

    def commit(self, files, message="change"):
        for rel, text in files.items():
            full = os.path.join(self.path, rel)
            if text is None:
                os.remove(full)
                continue
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh:
                fh.write(text)
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", message)
        return self._git("rev-parse", "HEAD")


BASE_TREE = {
    "docs/guide.md": "# Guide\n\nText.\n",
    "hw/rtl/interfaces/VX_demo_if.sv": IFACE,
    "hw/rtl/VX_gpu_pkg.sv": PKG,
    "hw/rtl/core/VX_unit.sv": UNIT,
    "sim/simx/unit.cpp": SIMX,
    "sw/runtime/include/vortex.h": HEADER,
    "sw/common/vx_gfx_abi.h": ABI,
    "sw/common/util.h": ABI.replace("rast_prim_t", "pair_t"),
    "VX_types.toml": TYPES,
    "ci/testcases/demo.yaml": TESTCASES,
    "ci/baselines/perf/demo.json": '{"demo:sgemm:rtlsim": {"cycles": 1000}}\n',
    ".github/workflows/demo.yml": WORKFLOW,
    ".github/workflows/later.yml": WORKFLOW_PUSH_FIRST,
    "VERSION": "VORTEX_VERSION=3.0\nTOOLCHAIN_REV=v3.0.1\nFIRESIM_REV=abc\n",
    "hw/rtl/VX_define.vh": DEFINE,
    "hw/rtl/core/VX_decode.sv": DECODE_SV,
    "sim/simx/decode.cpp": DECODE_CPP,
    "sw/kernel/include/vx_intrinsics.h": INTRINSICS,
    "VX_config.toml": CONFIG,
    "configure": CONFIGURE,
    "ci/gen_config.py": GEN_CONFIG,
    "Makefile.in": MAKEFILE_IN,
    "sw/kernel/Makefile": KERNEL_MK,
    "tests/demo/Makefile": KERNEL_MK,
}


@pytest.fixture
def scratch(tmp_path):
    repo = Scratch(tmp_path)
    base = repo.commit(BASE_TREE, "base")
    return repo, base


def run(scratch, files, message="change", event="pull_request"):
    repo, base = scratch
    head = repo.commit(files, message)
    r = ci.Repo(repo.path)
    try:
        return ci.classify(r, POLICY, base, head, event)
    finally:
        r.close()


def detectors(result):
    return {f["detector"] for f in result["contracts"]}


def checks(result, severity=None):
    return {e["check"] for e in result["evidence"] if severity in (None, e["severity"])}


def details(result, check):
    return " ".join(e["detail"] for e in result["evidence"] if e["check"] == check)


# --------------------------------------------------------------------------- #
# Policy and loader
# --------------------------------------------------------------------------- #

def test_repo_policy_lints_clean():
    assert ci.lint_policy(POLICY) == []
    assert ci.main(["lint-policy"]) == 0


def test_loader_matches_pyyaml():
    yaml = pytest.importorskip("yaml")
    with open(ci.DEFAULT_POLICY) as fh:
        text = fh.read()
    assert ci.load_yaml_subset(text) == yaml.safe_load(text)


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["path_rules"][0].update({"class": "R3"}), "may not exceed R2"),
    (lambda p: p["detectors"][0].update({"kind": "nope"}), "unknown kind"),
    (lambda p: p["evidence"][0].update({"min_class": "R3"}), "may not exceed R2"),
    (lambda p: p["path_rules"].append(dict(p["path_rules"][0])), "duplicate id"),
    (lambda p: p.pop("classes"), "classes must define"),
])
def test_lint_rejects_bad_policy(mutate, message):
    policy = json.loads(json.dumps(POLICY))
    mutate(policy)
    assert any(message in e for e in ci.lint_policy(policy))


@pytest.mark.parametrize("text", ["a: {b: 1}\n", "a: |\n  x\n", "a: [1, [2]]\n", "a:\n\tb: 1\n"])
def test_loader_rejects_unsupported_yaml(text):
    with pytest.raises(ci.PolicyError):
        ci.load_yaml_subset(text)


def test_globs():
    assert ci.path_matches("hw/rtl/core/VX_unit.sv", ["hw/rtl/**"])
    assert ci.path_matches("README.md", ["**/*.md"])
    assert ci.path_matches("hw/rtl/interfaces/VX_a_if.sv", ["hw/rtl/interfaces/*.sv"])
    assert not ci.path_matches("hw/rtl/interfaces/sub/VX_a_if.sv", ["hw/rtl/interfaces/*.sv"])
    assert not ci.path_matches("hw/rtlx/a.sv", ["hw/rtl/**"])


# --------------------------------------------------------------------------- #
# Path classes and cosmetic diffs
# --------------------------------------------------------------------------- #

def test_docs_only_is_r0(scratch):
    r = run(scratch, {"docs/guide.md": "# Guide\n\nBetter text.\n"})
    assert r["class"] == "R0"


def test_rtl_comment_only_is_cosmetic(scratch):
    r = run(scratch, {"hw/rtl/core/VX_unit.sv": UNIT.replace(
        "    reg [3:0] count_r;", "    // free-running\n    reg [3:0]   count_r;")})
    assert r["class"] == "R0"
    assert r["files"][0]["cosmetic"]
    assert "simx-rtl-lockstep" not in checks(r)


@pytest.mark.parametrize("path, old, new", [
    ("Makefile", "all:\n\tcp a b\n", "all:\n    cp a b\n"),
    ("ci/run.sh", '[ -n "$a" ] && echo\n', '[-n "$a"] && echo\n'),
    ("sim/simx/unit.cpp", "x = a++ + b;\n", "x = a + ++b;\n"),
    ("sw/common/util.h", "#define VX_F(x) x\n", "#define VX_F (x) x\n"),
    ("hw/rtl/core/VX_unit.sv", "assign y = a - -b;\n", "assign y = a --b;\n"),
    ("sim/simx/unit.cpp", 'puts("a  b");\n', 'puts("a b");\n'),
])
def test_token_boundaries_are_not_cosmetic(path, old, new):
    lang = ci.language(path)
    assert ci.normalize(old, lang) != ci.normalize(new, lang)


def test_alignment_is_cosmetic():
    lang = ci.language("sim/simx/unit.cpp")
    assert ci.normalize("int  a = 1;   // x\n\n", lang) == ci.normalize("  int a = 1;\n", lang)


def test_recipe_tab_to_spaces_is_not_r0(scratch):
    r = run(scratch, {"Makefile.in": MAKEFILE_IN.replace("\tcp -r", "    cp -r")})
    assert r["class"] != "R0"
    assert not r["files"][0]["cosmetic"]


def test_unterminated_comment_scans_in_linear_time(scratch):
    tail = "/* " * ((ci.MAX_BLOB - len(HEADER)) // 3)
    start = time.monotonic()
    r = run(scratch, {"sw/runtime/include/vortex.h": HEADER + tail,
                      "hw/rtl/interfaces/VX_demo_if.sv": IFACE + tail})
    assert time.monotonic() - start < 30
    assert not any(f["cosmetic"] for f in r["files"])


def test_python_indent_is_not_cosmetic(scratch):
    repo, _ = scratch
    base = repo.commit({"ci/tool.py": "if x:\n    a()\n    b()\n"})
    head = repo.commit({"ci/tool.py": "if x:\n    a()\nb()\n"})
    r = ci.classify(ci.Repo(repo.path), POLICY, base, head)
    assert not r["files"][0]["cosmetic"]


# --------------------------------------------------------------------------- #
# SimX/RTL lockstep
# --------------------------------------------------------------------------- #

def test_rtl_logic_without_simx_warns(scratch):
    r = run(scratch, {"hw/rtl/core/VX_unit.sv": UNIT.replace("count_r + 1", "count_r + 2")})
    assert r["class"] == "R2"
    assert "RTL logic changed with no SimX change" in details(r, "simx-rtl-lockstep")


def test_simx_without_rtl_warns(scratch):
    r = run(scratch, {"sim/simx/unit.cpp": SIMX.replace("delay_ = 2", "delay_ = 3")})
    assert "SimX model changed with no RTL change" in details(r, "simx-rtl-lockstep")


def test_both_sides_do_not_warn(scratch):
    r = run(scratch, {"hw/rtl/core/VX_unit.sv": UNIT.replace("count_r + 1", "count_r + 2"),
                      "sim/simx/unit.cpp": SIMX.replace("delay_ = 2", "delay_ = 3")})
    assert "simx-rtl-lockstep" not in checks(r)


def test_trace_only_rtl_change_does_not_warn(scratch):
    traced = UNIT.replace("            count_r <= count_r + 1;",
                          "            count_r <= count_r + 1;\n"
                          "            `TRACE(2, (\"%t: count=%0d (x)\\n\", $time, count_r))")
    r = run(scratch, {"hw/rtl/core/VX_unit.sv": traced})
    assert "simx-rtl-lockstep" not in checks(r)


# --------------------------------------------------------------------------- #
# Contract detectors
# --------------------------------------------------------------------------- #

def test_interface_signal_added_is_r3(scratch):
    iface = IFACE.replace("    logic  ready;", "    logic  ready;\n    logic  flush;").replace(
        "        input  ready\n    );", "        input  ready,\n        output flush\n    );", 1)
    r = run(scratch, {"hw/rtl/interfaces/VX_demo_if.sv": iface})
    assert r["class"] == "R3"
    assert "+signal flush" in r["contracts"][0]["detail"]
    assert "modport master: +output flush" in r["contracts"][0]["detail"]


def test_interface_direction_flip_is_r3(scratch):
    iface = IFACE.replace("        output data,\n        input  ready", "        input  data,\n        input  ready")
    r = run(scratch, {"hw/rtl/interfaces/VX_demo_if.sv": iface})
    assert "rtl-interface" in detectors(r)


def test_interface_parameter_without_port_list_is_r3(scratch):
    repo, _ = scratch
    iface = IFACE.replace(") ();", ");")
    base = repo.commit({"hw/rtl/interfaces/VX_demo_if.sv": iface})
    head = repo.commit({"hw/rtl/interfaces/VX_demo_if.sv": iface.replace(
        "parameter type data_t = logic", "parameter type data_t = logic,\n    parameter int WIDTH = 32")})
    r = ci.classify(ci.Repo(repo.path), POLICY, base, head)
    assert r["class"] == "R3"
    assert r["contracts"][0]["detail"] == "+param WIDTH"


def test_interface_comment_is_not_r3(scratch):
    r = run(scratch, {"hw/rtl/interfaces/VX_demo_if.sv": IFACE.replace(
        "    logic  valid;", "    // asserted for one cycle\n    logic  valid;")})
    assert r["class"] == "R0"


def test_new_interface_is_r3(scratch):
    r = run(scratch, {"hw/rtl/interfaces/VX_new_if.sv": IFACE.replace("VX_demo_if", "VX_new_if")})
    assert r["class"] == "R3"
    assert "interface added: 3 signals; modports master, slave" in r["contracts"][0]["detail"]


def test_pkg_struct_added_is_r3(scratch):
    r = run(scratch, {"hw/rtl/VX_gpu_pkg.sv": PKG.replace(
        "endpackage", "    typedef struct packed {\n        logic [7:0] id;\n    } rsp_t;\n\nendpackage")})
    assert r["class"] == "R3"
    assert "typedefs added: rsp_t" in r["contracts"][0]["detail"]


def test_pkg_isa_localparam_added_is_r3(scratch):
    r = run(scratch, {"hw/rtl/VX_gpu_pkg.sv": PKG.replace(
        "    localparam INST_ALU_SUB = 4'b0001;",
        "    localparam INST_ALU_SUB = 4'b0001;\n    localparam INST_ALU_XOR = 4'b0010;")})
    assert "isa-localparams" in detectors(r)
    assert "encodings added: INST_ALU_XOR" in r["contracts"][0]["detail"]


def test_pkg_isa_localparam_changed_is_r3(scratch):
    r = run(scratch, {"hw/rtl/VX_gpu_pkg.sv": PKG.replace("INST_ALU_SUB = 4'b0001", "INST_ALU_SUB = 4'b0011")})
    assert "encodings changed: INST_ALU_SUB" in r["contracts"][0]["detail"]


def test_define_isa_macro_added_is_r3(scratch):
    r = run(scratch, {"hw/rtl/VX_define.vh": DEFINE.replace(
        "`define NUM_REGS", "`define INST_FENCE_I 1'b1\n`define NUM_REGS")})
    assert "isa-localparams" in detectors(r)


def test_define_non_isa_macro_is_r2(scratch):
    r = run(scratch, {"hw/rtl/VX_define.vh": DEFINE.replace("NUM_REGS 32", "NUM_REGS 64")})
    assert r["class"] == "R2"


def test_rtl_decoder_new_op_is_r3(scratch):
    r = run(scratch, {"hw/rtl/core/VX_decode.sv": DECODE_SV.replace(
        "            default:", "            3'h4: r_type = INST_ALU_XOR;\n            default:")})
    assert r["class"] == "R3"
    assert "decodes new: INST_ALU_XOR" in r["contracts"][0]["detail"]


def test_rtl_decoder_restructure_is_r2(scratch):
    r = run(scratch, {"hw/rtl/core/VX_decode.sv": DECODE_SV.replace("case (funct3)", "unique case (funct3)")})
    assert r["class"] == "R2"


def test_simx_decoder_new_op_is_r3(scratch):
    r = run(scratch, {"sim/simx/decode.cpp": DECODE_CPP.replace(
        "funct3 ? AluType::SUB", "(funct3 == 4) ? AluType::XOR : funct3 ? AluType::SUB")})
    assert "decodes new: AluType::XOR" in r["contracts"][0]["detail"]


def test_new_intrinsic_is_r3(scratch):
    r = run(scratch, {"sw/kernel/include/vx_intrinsics.h": INTRINSICS +
                      "\ninline void vx_barrier_sync() {\n"
                      "  __asm__ volatile (\".insn r 0x0b, 4, 0, x0, x0, x0\");\n}\n"})
    assert r["class"] == "R3"
    assert "added: vx_barrier_sync" in r["contracts"][0]["detail"]


def test_changed_intrinsic_encoding_is_r3(scratch):
    r = run(scratch, {"sw/kernel/include/vx_intrinsics.h": INTRINSICS.replace("0x0b, 0, 0", "0x0b, 0, 1")})
    assert r["class"] == "R3"
    assert "definition changed: vx_fence" in r["contracts"][0]["detail"]


def test_removed_interface_is_r3(scratch):
    r = run(scratch, {"hw/rtl/interfaces/VX_demo_if.sv": None})
    assert "rtl-interface" in detectors(r)


def test_pkg_struct_change_is_r3(scratch):
    r = run(scratch, {"hw/rtl/VX_gpu_pkg.sv": PKG.replace("logic [3:0] op;", "logic [4:0] op;")})
    assert r["class"] == "R3"
    assert "req_t" in r["contracts"][0]["detail"]


def test_pkg_localparam_change_is_r2(scratch):
    r = run(scratch, {"hw/rtl/VX_gpu_pkg.sv": PKG.replace("NUM_LANES = 4", "NUM_LANES = 8")})
    assert r["class"] == "R2"
    assert not r["contracts"]


def test_types_toml_value_change_is_r3(scratch):
    r = run(scratch, {"VX_types.toml": TYPES.replace("0x7C0", "0x7C1")})
    assert r["class"] == "R3"
    assert "VX_CSR_FOO: 0x7C0 -> 0x7C1" in r["contracts"][0]["detail"]


def test_types_toml_addition_is_r3(scratch):
    r = run(scratch, {"VX_types.toml": TYPES + "VX_CSR_BAR = 0x7C2\n"})
    assert r["class"] == "R3"
    assert "added: csr_base.VX_CSR_BAR" in r["contracts"][0]["detail"]


def test_types_toml_key_moved_between_sections_is_r2(scratch):
    moved = TYPES.replace("VX_CSR_FOO = 0x7C0\n", "") + "\n[csr_more]\nVX_CSR_FOO = 0x7C0\n"
    r = run(scratch, {"VX_types.toml": moved})
    assert not r["contracts"]


def test_abi_struct_change_is_r3(scratch):
    r = run(scratch, {"sw/common/vx_gfx_abi.h": ABI.replace("uint32_t y;", "uint32_t y;\n  uint32_t z;")})
    assert r["class"] == "R3"
    assert "rast_prim_t" in r["contracts"][0]["detail"]


def test_non_abi_struct_change_is_r2(scratch):
    r = run(scratch, {"sw/common/util.h": ABI.replace("rast_prim_t", "pair_t").replace(
        "uint32_t y;", "uint32_t y;\n  uint32_t z;")})
    assert r["class"] == "R2"


def test_removed_installed_symbol_is_r3(scratch):
    r = run(scratch, {"sw/runtime/include/vortex.h": HEADER.replace(
        "int vx_dev_close(vx_device_h hdevice);\n", "")})
    assert r["class"] == "R3"
    assert "removed: vx_dev_close" in r["contracts"][0]["detail"]


def test_changed_runtime_prototype_is_r3(scratch):
    r = run(scratch, {"sw/runtime/include/vortex.h": HEADER.replace(
        "int vx_dev_close(vx_device_h hdevice);", "int vx_dev_close(vx_device_h hdevice, int flags);")})
    assert "prototype changed: vx_dev_close" in r["contracts"][0]["detail"]


def test_added_attribute_is_not_a_removal(scratch):
    r = run(scratch, {"sw/runtime/include/vortex.h": HEADER.replace(
        "int vx_printf(const char * format, ...);",
        "int vx_printf(const char * format, ...) __attribute__((format(printf, 1, 2)));")})
    assert r["class"] == "R2"


def test_symbol_moved_between_installed_headers_is_r3(scratch):
    r = run(scratch, {
        "sw/runtime/include/vortex.h": HEADER.replace("int vx_dev_close(vx_device_h hdevice);\n", ""),
        "sw/runtime/include/vortex_ext.h": "#pragma once\nint vx_dev_close(void* hdevice);\n"})
    text = " ".join(f["detail"] for f in r["contracts"])
    assert "removed: vx_dev_close" in text and "added: vx_dev_close" in text


def test_added_symbol_is_r3(scratch):
    r = run(scratch, {"sw/runtime/include/vortex.h": HEADER.replace(
        "int vx_printf", "int vx_dev_reset(vx_device_h hdevice);\nint vx_printf")})
    assert r["class"] == "R3"
    assert r["contracts"][0]["detail"] == "added: vx_dev_reset"


def test_installed_macro_value_change_is_r3(scratch):
    r = run(scratch, {"sw/runtime/include/vortex.h": HEADER.replace("VX_DEMO_FLAG 1", "VX_DEMO_FLAG 2")})
    assert r["contracts"][0]["detail"] == "macro value changed: VX_DEMO_FLAG"


def test_installed_enum_change_is_r3(scratch):
    repo, _ = scratch
    with_enum = HEADER.replace("typedef void* vx_device_h;",
                               "typedef enum { VX_MEM_READ = 1, VX_MEM_WRITE = 2 } vx_mem_flags_t;\n"
                               "typedef void* vx_device_h;")
    base = repo.commit({"sw/runtime/include/vortex.h": with_enum})
    head = repo.commit({"sw/runtime/include/vortex.h": with_enum.replace("VX_MEM_WRITE = 2", "VX_MEM_WRITE = 4")})
    r = ci.classify(ci.Repo(repo.path), POLICY, base, head)
    assert "layout changed: vx_mem_flags_t" in r["contracts"][0]["detail"]


def test_abi_constant_added_is_r3(scratch):
    r = run(scratch, {"sw/common/vx_gfx_abi.h": ABI + "\n#define VX_GFX_MAX_VARYINGS 12\n"})
    assert r["class"] == "R3"
    assert "added: VX_GFX_MAX_VARYINGS" in r["contracts"][0]["detail"]


def test_function_becoming_a_template_is_not_removed(scratch):
    r = run(scratch, {"sw/runtime/include/vortex.h": HEADER.replace(
        "int vx_printf(const char * format, ...);",
        "#ifdef __cplusplus\ntemplate <int FD = 28, int OFF = 0>\ninline int vx_printf(const char* format) { return FD + OFF; }\n#endif")})
    assert all("removed" not in f["detail"] for f in r["contracts"])


def test_object_like_macro_becoming_function_like_is_r3(scratch):
    repo, _ = scratch
    header = HEADER.replace("#define VX_DEMO_FLAG 1", "#define VX_F (x) ((x) + 1)")
    base = repo.commit({"sw/runtime/include/vortex.h": header})
    head = repo.commit({"sw/runtime/include/vortex.h": header.replace("VX_F (x)", "VX_F(x)")})
    r = ci.classify(ci.Repo(repo.path), POLICY, base, head)
    assert r["contracts"][0]["detail"] == "macro value changed: VX_F"


def test_newline_path_does_not_hide_a_header_change(scratch):
    repo, _ = scratch
    base = repo.commit({"docs/\nz.md": "a\n"})
    head = repo.commit({"docs/\nz.md": "b\n", "sw/runtime/include/vortex.h": HEADER.replace(
        "int vx_dev_close(vx_device_h hdevice);", "int vx_dev_close(vx_device_h hdevice, int flags);")})
    r = ci.classify(ci.Repo(repo.path), POLICY, base, head)
    assert r["class"] == "R3"
    assert "installed-api" in detectors(r)



def test_multiline_macro_does_not_hide_declarations(scratch):
    macro = "#define VX_ISSUE(a) do {   \\\n    asm volatile (\"x\" : : \"r\"(a)); \\\n  } while (0)\n\n"
    r = run(scratch, {"sw/runtime/include/vortex.h": HEADER.replace(
        "typedef void* vx_device_h;", macro + "typedef void* vx_device_h;")})
    assert [f["detail"] for f in r["contracts"]] == ["added: VX_ISSUE"]


# --------------------------------------------------------------------------- #
# Workflow trust boundary
# --------------------------------------------------------------------------- #

def test_pull_request_target_is_r3(scratch):
    r = run(scratch, {".github/workflows/demo.yml": WORKFLOW.replace("  pull_request:", "  pull_request_target:")})
    assert "workflow-trust" in detectors(r)


def test_self_hosted_on_pull_request_is_r3(scratch):
    r = run(scratch, {".github/workflows/demo.yml": WORKFLOW.replace(
        "runs-on: ubuntu-22.04", "runs-on: [self-hosted, linux, vivado]")})
    assert r["class"] == "R3"
    assert "self-hosted runner reachable" in r["contracts"][0]["detail"]


def test_dispatch_guarded_self_hosted_is_r2(scratch):
    r = run(scratch, {".github/workflows/demo.yml": WORKFLOW.replace(
        "    runs-on: ubuntu-22.04", "    if: ${{ inputs.gpu }}\n    runs-on: [self-hosted, linux]")})
    assert r["class"] == "R2"


def test_event_text_in_script_is_r3(scratch):
    r = run(scratch, {".github/workflows/demo.yml": WORKFLOW.replace(
        "echo build", "echo \"${{ github.event.pull_request.title }}\"")})
    assert "interpolated into a script" in r["contracts"][0]["detail"]


def test_self_hosted_with_pull_request_not_first_is_r3(scratch):
    r = run(scratch, {".github/workflows/later.yml": WORKFLOW_PUSH_FIRST.replace(
        "runs-on: ubuntu-22.04", "runs-on: [self-hosted, linux]")})
    assert r["class"] == "R3"
    assert "self-hosted runner reachable" in r["contracts"][0]["detail"]


def test_pull_request_target_not_first_is_r3(scratch):
    r = run(scratch, {".github/workflows/later.yml": WORKFLOW_PUSH_FIRST.replace(
        "  pull_request:", "  pull_request_target:")})
    assert "trigger pull_request_target" in r["contracts"][0]["detail"]


@pytest.mark.parametrize("on", [
    "'on':\n  push:\n  pull_request:\n",
    "\"on\":\n  push:\n  pull_request:\n",
    "on:\n  - push\n  - pull_request\n",
    "on:\n- push\n- pull_request\n",
    "on: [push, pull_request]\n",
])
def test_trigger_forms(on):
    assert ci._workflow_triggers(on + "jobs:\n  a:\n    runs-on: x\n") == {"push", "pull_request"}


def test_disjunction_with_pull_request_is_not_a_guard(scratch):
    r = run(scratch, {".github/workflows/later.yml": WORKFLOW_PUSH_FIRST.replace(
        "    runs-on: ubuntu-22.04",
        "    if: github.event_name == 'push' || github.event_name == 'pull_request'\n"
        "    runs-on: [self-hosted, linux]")})
    assert r["class"] == "R3"


def test_fork_guard_keeps_self_hosted_r2(scratch):
    r = run(scratch, {".github/workflows/later.yml": WORKFLOW_PUSH_FIRST.replace(
        "    runs-on: ubuntu-22.04",
        "    if: github.event_name == 'push' || (github.event_name == 'pull_request' && "
        "github.event.pull_request.head.repo.full_name == github.repository)\n"
        "    runs-on: [self-hosted, linux]")})
    assert r["class"] == "R2"


def test_event_text_in_github_script_is_r3(scratch):
    r = run(scratch, {".github/workflows/demo.yml": WORKFLOW.replace(
        "      - run: echo build",
        "      - uses: actions/github-script@v7\n        with:\n          script: |\n"
        "            core.info(\"${{ github.event.pull_request.title }}\")")})
    assert "interpolated into a script" in r["contracts"][0]["detail"]


def test_workflow_run_text_in_script_is_r3(scratch):
    r = run(scratch, {".github/workflows/demo.yml": WORKFLOW.replace(
        "echo build", "echo \"${{ github.event.workflow_run.head_branch }}\"")})
    assert "interpolated into a script" in r["contracts"][0]["detail"]


def test_event_text_after_hash_in_run_block_is_r3(scratch):
    r = run(scratch, {".github/workflows/later.yml": WORKFLOW_PUSH_FIRST.replace(
        "          echo build", "          echo build  # ${{ github.event.pull_request.body }}")})
    assert "interpolated into a script" in r["contracts"][0]["detail"]


def test_workflow_comment_is_r0(scratch):
    r = run(scratch, {".github/workflows/later.yml": WORKFLOW_PUSH_FIRST.replace(
        "jobs:", "# one job\njobs:  # all of them")})
    assert r["class"] == "R0"


def test_event_text_in_env_is_r2(scratch):
    r = run(scratch, {".github/workflows/demo.yml": WORKFLOW.replace(
        "      - run: echo build",
        "      - env:\n          TITLE: ${{ github.event.pull_request.title }}\n        run: echo \"$TITLE\"")})
    assert r["class"] == "R2"


# --------------------------------------------------------------------------- #
# Evidence: silenced tests, golden data, toolchain pin
# --------------------------------------------------------------------------- #

def test_known_issue_added_warns(scratch):
    r = run(scratch, {"ci/testcases/demo.yaml": TESTCASES.replace(
        "    drivers: [simx, rtlsim]", "    drivers: [simx, rtlsim]\n    known_issue: \"hangs on rtlsim\"")})
    assert r["class"] == "R2"
    assert "known_issue added" in details(r, "test-silencing")


def test_tolerance_widened_warns(scratch):
    r = run(scratch, {"ci/testcases/demo.yaml": TESTCASES.replace("tolerance: 0.03", "tolerance: 0.08")})
    assert "tolerance widened 0.03 -> 0.08" in details(r, "test-silencing")


def test_tolerance_tightened_is_quiet(scratch):
    r = run(scratch, {"ci/testcases/demo.yaml": TESTCASES.replace("tolerance: 0.03", "tolerance: 0.02")})
    assert "test-silencing" not in checks(r)
    assert r["class"] == "R1"


def test_args_shrunk_warns(scratch):
    r = run(scratch, {"ci/testcases/demo.yaml": TESTCASES.replace('"-n64 -m32"', '"-n16 -m32"')})
    assert "args shrunk [64, 32] -> [16, 32]" in details(r, "test-silencing")


def test_diff_attribute_cannot_hide_silencing(scratch):
    r = run(scratch, {".gitattributes": "*.yaml -diff\n",
                      "ci/testcases/demo.yaml": TESTCASES.replace("tolerance: 0.03", "tolerance: 0.08")
                      .replace('"-n64 -m32"', '"-n16 -m32"')})
    text = details(r, "test-silencing")
    assert "tolerance widened 0.03 -> 0.08" in text
    assert "args shrunk [64, 32] -> [16, 32]" in text


def test_args_grown_is_quiet(scratch):
    r = run(scratch, {"ci/testcases/demo.yaml": TESTCASES.replace('"-n64 -m32"', '"-n128 -m32"')})
    assert "test-silencing" not in checks(r)


def test_case_removed_warns(scratch):
    r = run(scratch, {"ci/testcases/demo.yaml": TESTCASES.split("  - id: vecadd")[0]})
    assert "cases removed: vecadd" in details(r, "test-silencing")


def test_hand_edited_baseline_warns(scratch):
    r = run(scratch, {"ci/baselines/perf/demo.json": '{"demo:sgemm:rtlsim": {"cycles": 1100}}\n'},
            message="bump the sgemm number")
    assert r["class"] == "R2"
    text = details(r, "golden-provenance")
    assert "no commit message names how it was regenerated" in text
    assert "with no code change" in text


def test_regenerated_baseline_is_info(scratch):
    r = run(scratch, {"ci/baselines/perf/demo.json": '{"demo:sgemm:rtlsim": {"cycles": 900}}\n',
                      "hw/rtl/core/VX_unit.sv": UNIT.replace("count_r + 1", "count_r + 2"),
                      "sim/simx/unit.cpp": SIMX.replace("delay_ = 2", "delay_ = 3")},
            message="unit: count by two\n\npytest ci -m perf_gate --update-baselines")
    assert checks(r, "warn") == set()
    assert "golden-provenance" in checks(r, "info")


def test_toolchain_pin_is_r3(scratch):
    r = run(scratch, {"VERSION": "VORTEX_VERSION=3.0\nTOOLCHAIN_REV=v3.0.2\nFIRESIM_REV=abc\n"})
    assert r["class"] == "R3"
    assert r["contracts"][0]["detail"] == "TOOLCHAIN_REV v3.0.1 -> v3.0.2"
    assert "TOOLCHAIN_REV v3.0.1 -> v3.0.2" in details(r, "toolchain-pin")


def test_firesim_pin_is_r3(scratch):
    r = run(scratch, {"VERSION": "VORTEX_VERSION=3.0\nTOOLCHAIN_REV=v3.0.1\nFIRESIM_REV=def\n"})
    assert "toolchain-pins" in detectors(r)


def test_version_comment_is_r0(scratch):
    r = run(scratch, {"VERSION": "# pins\nVORTEX_VERSION=3.0\nTOOLCHAIN_REV=v3.0.1\nFIRESIM_REV=abc\n"})
    assert r["class"] == "R0"


# --------------------------------------------------------------------------- #
# Configuration knobs
# --------------------------------------------------------------------------- #

def test_config_knob_added_is_r3(scratch):
    r = run(scratch, {"VX_config.toml": CONFIG + "VX_CFG_FETCH_WIDTH = 1\n"})
    assert r["class"] == "R3"
    assert "added: base.VX_CFG_FETCH_WIDTH" in r["contracts"][0]["detail"]


def test_config_default_changed_is_r3(scratch):
    r = run(scratch, {"VX_config.toml": CONFIG.replace("VX_CFG_NUM_WARPS = 4", "VX_CFG_NUM_WARPS = 8")})
    assert "changed base.VX_CFG_NUM_WARPS: 4 -> 8" in r["contracts"][0]["detail"]


def test_config_derivation_changed_is_r3(scratch):
    r = run(scratch, {"VX_config.toml": CONFIG.replace("/ 16", "/ 8")})
    assert "config-knobs" in detectors(r)


def test_config_knob_removed_is_r3(scratch):
    r = run(scratch, {"VX_config.toml": CONFIG.replace("VX_CFG_NUM_THREADS = 4\n", "")})
    assert "removed: base.VX_CFG_NUM_THREADS" in r["contracts"][0]["detail"]


def test_config_comment_is_r0(scratch):
    r = run(scratch, {"VX_config.toml": "# core shape\n" + CONFIG})
    assert r["class"] == "R0"


def test_configure_knob_line_is_r3(scratch):
    r = run(scratch, {"configure": CONFIGURE.replace(
        'CONFIGS="-DVX_CFG_XLEN=$default_xlen"', 'CONFIGS="-DVX_CFG_XLEN=$default_xlen -DVX_CFG_NUM_CORES=1"')})
    assert r["class"] == "R3"
    assert "knobs +: VX_CFG_NUM_CORES" in r["contracts"][0]["detail"]


def test_configure_other_edit_is_r2(scratch):
    r = run(scratch, {"configure": CONFIGURE.replace("echo done", "echo configured")})
    assert r["class"] == "R2"


def test_gen_config_logic_is_r3(scratch):
    r = run(scratch, {"ci/gen_config.py": GEN_CONFIG.replace("return value", "return value.strip()")})
    assert "config-knobs" in detectors(r)


def test_gen_config_comment_is_r0(scratch):
    r = run(scratch, {"ci/gen_config.py": GEN_CONFIG.replace("def resolve", "# expressions\ndef resolve")})
    assert r["class"] == "R0"


# --------------------------------------------------------------------------- #
# Build surface
# --------------------------------------------------------------------------- #

def test_install_list_change_is_r3(scratch):
    r = run(scratch, {"Makefile.in": MAKEFILE_IN.replace("gfx_dcr.h", "gfx_dcr.h rtu_cfg.h")})
    assert r["class"] == "R3"
    assert "install lines changed" in r["contracts"][0]["detail"]


def test_install_rule_added_is_r3(scratch):
    r = run(scratch, {"Makefile.in": MAKEFILE_IN + "\nuninstall:\n\trm -rf $(KERNEL_INC_DST)\n"})
    assert "build-surface" in detectors(r)


def test_abi_flag_change_is_r3(scratch):
    r = run(scratch, {"sw/kernel/Makefile": KERNEL_MK.replace("rv32imaf", "rv32imafc")})
    assert "abi lines changed" in r["contracts"][0]["detail"]


def test_abi_flag_added_is_r3(scratch):
    r = run(scratch, {"sw/kernel/Makefile": KERNEL_MK + "CFLAGS += -Xclang -target-feature -Xclang +xvortexldstwide\n"})
    assert "abi lines changed" in r["contracts"][0]["detail"]


def test_exported_toolchain_path_change_is_r3(scratch):
    r = run(scratch, {"sw/kernel/Makefile": KERNEL_MK.replace("$(HOME)/tools", "/opt/vortex/tools")})
    assert "toolchain lines changed" in r["contracts"][0]["detail"]


def test_other_makefile_edit_is_r2(scratch):
    r = run(scratch, {"sw/kernel/Makefile": KERNEL_MK.replace("-O3", "-O2")})
    assert r["class"] == "R2"


def test_test_makefile_flag_is_not_r3(scratch):
    r = run(scratch, {"tests/demo/Makefile": KERNEL_MK.replace("rv32imaf", "rv32imafc")})
    assert r["class"] in ("R1", "R2")


# --------------------------------------------------------------------------- #
# Indeterminate diffs and output
# --------------------------------------------------------------------------- #

def test_push_without_base_fails_closed(scratch):
    repo, base = scratch
    r = ci.classify(ci.Repo(repo.path), POLICY, "0" * 40, base, "push")
    assert r["indeterminate"] and r["class"] == "R3"


def test_pull_request_without_base_is_r2(scratch):
    repo, base = scratch
    r = ci.classify(ci.Repo(repo.path), POLICY, "deadbeef", base, "pull_request")
    assert r["indeterminate"] and r["class"] == "R2"


def test_root_commit_replays_as_indeterminate(scratch):
    repo, base = scratch
    r = ci.classify_commit(ci.Repo(repo.path), POLICY, base)
    assert r["indeterminate"] and r["class"] == POLICY["indeterminate"]["replay"]


def test_renderers(scratch):
    r = run(scratch, {"hw/rtl/VX_gpu_pkg.sv": PKG.replace("logic [3:0] op;", "logic [4:0] op;")})
    md = ci.render_markdown(r)
    assert md.startswith("### Change impact: R3")
    assert "gpu-pkg-types" in md
    assert "R3" in ci.render_text(r, verbose=True)
    json.dumps(r)


def test_cli_classify_json(scratch, capsys):
    repo, base = scratch
    head = repo.commit({"docs/guide.md": "# Guide\n\nNew.\n"})
    assert ci.main(["--repo", repo.path, "classify", "--base", base, "--head", head,
                    "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["class"] == "R0"


# --------------------------------------------------------------------------- #
# Replay over real history
# --------------------------------------------------------------------------- #

REPLAY = [
    # sha, class, detectors that must fire
    ("63c381a9b", "R3", {"rtl-interface", "gpu-pkg-types"}),   # VX_sched_csr_if loses a signal
    ("fff614f17", "R3", {"isa-encoding"}),                      # OM CSRs removed
    ("5612c9e36", "R3", {"sw-common-abi"}),                     # rast_prim_t grows
    ("942e381aa", "R3", {"config-knobs"}),                      # memory-model knobs added
    ("6f372b0ae", "R3", {"isa-encoding", "installed-api"}),     # RT hit bits, DXA issue macros
    ("8311a7259", "R2", set()),                                 # typedefs reordered, not changed
    ("cc2f1a501", "R1", set()),                                 # design documents
]


def _have(sha):
    return subprocess.run(["git", "-C", ci.ROOT, "cat-file", "-e", sha + "^{commit}"],
                          stderr=subprocess.DEVNULL).returncode == 0 and \
        subprocess.run(["git", "-C", ci.ROOT, "cat-file", "-e", sha + "^^{commit}"],
                       stderr=subprocess.DEVNULL).returncode == 0


@pytest.mark.parametrize("sha, cls, dets", REPLAY)
def test_replay_known_commits(sha, cls, dets):
    if not _have(sha):
        pytest.skip("{} not in this clone".format(sha))
    repo = ci.Repo(ci.ROOT)
    try:
        r = ci.classify_commit(repo, POLICY, sha)
    finally:
        repo.close()
    assert r["class"] == cls, r["reasons"]
    assert dets <= detectors(r)


def test_replay_r3_has_a_contract_on_master():
    """Every R3 among real commits names the external change that made it R3."""
    if subprocess.run(["git", "-C", ci.ROOT, "rev-parse", "--verify", "-q", "origin/master"],
                      stdout=subprocess.DEVNULL).returncode != 0:
        pytest.skip("origin/master not in this clone")
    shas = subprocess.run(["git", "-C", ci.ROOT, "rev-list", "--no-merges", "--max-count=60",
                           "origin/master"], stdout=subprocess.PIPE, text=True).stdout.split()
    if len(shas) < 60:
        pytest.skip("shallow clone")
    repo = ci.Repo(ci.ROOT)
    try:
        results = [ci.classify_commit(repo, POLICY, s) for s in shas]
    finally:
        repo.close()
    for r in results:
        assert (r["class"] == "R3") == bool(r["contracts"]), r["head"]
