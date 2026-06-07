#!/usr/bin/env python3
"""
DTM coverage tests for the simx debug stack.

Drives the remote-bitbang TCP server directly (no OpenOCD/GDB dependency),
exercising: the JTAG TAP state machine, IR=IDCODE, IR=DBUS DMI read/write,
DMSTATUS decoding, abstract-register read/write, abstract-memory read/write,
and resume-to-natural-completion via the emulator halt hook.

Test list (each prints its own PASS/FAIL line; exit code 0 iff all pass):
    1. IDCODE read returns 0xdeadbeef
    2. DMSTATUS initial state: authenticated=1, version=2, allhalted=1
    3. Abstract register read of PC returns STARTUP_ADDR
    4. Abstract register read of MISA: MXL matches XLEN, bit 8 (I) is set
    5. Abstract register write+read GPR round-trip
    6. Abstract memory write+read round-trip in a scratch region
    7. Resume via DMCONTROL.resumereq eventually halts again
       (Emulator.notify_program_completed path)
"""
import os
import socket
import subprocess
import sys
import time

RBB_PORT = int(os.environ.get("DTM_RBB_PORT", "19823"))
SIMX = os.environ.get("SIMX_BIN", "sim/simx/simx")
PROGRAM = os.environ.get("DTM_PROGRAM", "tests/kernel/fibonacci/fibonacci.vxbin")
XLEN = int(os.environ.get("DTM_XLEN", "32"))

STARTUP_ADDR = 0x80000000
SCRATCH_ADDR = 0x80100000  # well clear of fibonacci's text/data

# JTAG / DMI constants (must match sim/simx/dtm/jtag_dtm.cpp)
ABITS = 7
DR_BITS = ABITS + 34  # 41

IR_IDCODE = 0x01
IR_DBUS   = 0x11

# Debug Module register addresses (DMI space, 7-bit)
DM_DATA0      = 0x04
DM_DMCONTROL  = 0x10
DM_DMSTATUS   = 0x11
DM_COMMAND    = 0x17

# Abstract register addresses (see debug_module.cpp::read_register)
REG_GPR_BASE = 0x1000
REG_PC       = 0x1020
REG_MISA     = 0x0301


def pin(tck, tms, tdi):
    return bytes([ord('0') | (tck << 2) | (tms << 1) | tdi])


def clocks(sock, tms_bits, tdi_bits=None, read=False):
    if tdi_bits is None:
        tdi_bits = [0] * len(tms_bits)
    buf = bytearray()
    for tms, tdi in zip(tms_bits, tdi_bits):
        buf += pin(0, tms, tdi)
        if read:
            buf += b'R'
        buf += pin(1, tms, tdi)
    sock.sendall(bytes(buf))
    if not read:
        return None
    need = len(tms_bits)
    reply = b''
    while len(reply) < need:
        chunk = sock.recv(need - len(reply))
        if not chunk:
            raise RuntimeError("RBB socket closed")
        reply += chunk
    return [1 if b == ord('1') else 0 for b in reply]


def bits_of(value, nbits):
    return [(value >> i) & 1 for i in range(nbits)]


def int_of(bits):
    v = 0
    for i, b in enumerate(bits):
        v |= b << i
    return v


def tap_reset(sock):
    clocks(sock, [1, 1, 1, 1, 1, 0])  # -> Run-Test-Idle


def shift_ir(sock, value, nbits=5):
    clocks(sock, [1, 1, 0, 0])                  # RTI -> Shift-IR
    tms = [0] * (nbits - 1) + [1]
    clocks(sock, tms, bits_of(value, nbits))    # shift + Exit1-IR
    clocks(sock, [1, 0])                        # Update-IR -> RTI


def shift_dr(sock, value, nbits, read=False):
    clocks(sock, [1, 0, 0])                     # RTI -> Shift-DR
    tms = [0] * (nbits - 1) + [1]
    tdo = clocks(sock, tms, bits_of(value, nbits), read=read)
    clocks(sock, [1, 0])                        # Update-DR -> RTI
    return int_of(tdo) if read else None


def dmi_raw(sock, addr, data, op):
    """One DMI transaction. Returns (prev_result_data, status)."""
    dr = ((addr & ((1 << ABITS) - 1)) << 34) | ((data & 0xFFFFFFFF) << 2) | (op & 3)
    out = shift_dr(sock, dr, DR_BITS, read=True)
    return (out >> 2) & 0xFFFFFFFF, out & 0x3


def dmi_read(sock, addr):
    # 1st transaction queues the read; 2nd (nop) captures the result.
    dmi_raw(sock, addr, 0, 1)
    data, status = dmi_raw(sock, 0, 0, 0)
    if status != 0:
        raise RuntimeError(f"dmi_read(0x{addr:x}) status={status}")
    return data


def dmi_write(sock, addr, value):
    _, status = dmi_raw(sock, addr, value, 2)
    if status != 0:
        raise RuntimeError(f"dmi_write(0x{addr:x}, 0x{value:x}) status={status}")


def abstract_reg_read(sock, regaddr):
    """Issue Access Register (cmdtype=0, transfer=1, read). Return DATA0."""
    # aarsize=2 (32-bit) for XLEN=32, aarsize=3 (64-bit) for XLEN=64
    aarsize = 2 if XLEN == 32 else 3
    cmd = (0 << 24) | (aarsize << 20) | (1 << 17) | (regaddr & 0xFFFF)
    dmi_write(sock, DM_COMMAND, cmd)
    return dmi_read(sock, DM_DATA0)


def abstract_reg_write(sock, regaddr, value):
    aarsize = 2 if XLEN == 32 else 3
    dmi_write(sock, DM_DATA0, value & 0xFFFFFFFF)
    cmd = (0 << 24) | (aarsize << 20) | (1 << 17) | (1 << 16) | (regaddr & 0xFFFF)
    dmi_write(sock, DM_COMMAND, cmd)


def abstract_mem_write32(sock, addr, value):
    # Access Memory cmdtype=0x02; per DM code, address comes from DATA2
    # when DATA2/3 are nonzero. Write data goes in DATA0.
    dmi_write(sock, 0x5, 0)            # DATA1 (high data) = 0
    dmi_write(sock, 0x6, addr & 0xFFFFFFFF)  # DATA2 = addr
    dmi_write(sock, 0x7, 0)            # DATA3 (high addr) = 0
    dmi_write(sock, DM_DATA0, value & 0xFFFFFFFF)
    cmd = (0x02 << 24) | (2 << 20) | (1 << 16)  # aamsize=32-bit, write=1
    dmi_write(sock, DM_COMMAND, cmd)


def abstract_mem_read32(sock, addr):
    dmi_write(sock, 0x6, addr & 0xFFFFFFFF)  # DATA2 = addr
    dmi_write(sock, 0x7, 0)
    cmd = (0x02 << 24) | (2 << 20)           # aamsize=32-bit, write=0
    dmi_write(sock, DM_COMMAND, cmd)
    return dmi_read(sock, DM_DATA0)


# --- tests ---------------------------------------------------------------


def t_idcode(sock):
    tap_reset(sock)
    shift_ir(sock, IR_IDCODE)
    idcode = shift_dr(sock, 0, 32, read=True)
    expected = 0xDEADBEEF
    ok = idcode == expected
    return ok, f"IDCODE=0x{idcode:08x} (expected 0x{expected:08x})"


def t_dmstatus_initial(sock):
    shift_ir(sock, IR_DBUS)
    status = dmi_read(sock, DM_DMSTATUS)
    version = status & 0xF
    authenticated = (status >> 7) & 1
    allhalted = (status >> 9) & 1
    allrunning = (status >> 11) & 1
    ok = (version == 2 and authenticated == 1 and allhalted == 1 and allrunning == 0)
    return ok, (f"DMSTATUS=0x{status:08x} (ver={version}, auth={authenticated}, "
                f"allhalted={allhalted}, allrunning={allrunning})")


def t_abstract_read_pc(sock):
    pc = abstract_reg_read(sock, REG_PC)
    ok = pc == STARTUP_ADDR
    return ok, f"PC=0x{pc:08x} (expected 0x{STARTUP_ADDR:08x})"


def t_abstract_read_misa(sock):
    misa = abstract_reg_read(sock, REG_MISA)
    mxl = (misa >> 30) & 0x3
    i_ext = (misa >> 8) & 1
    expected_mxl = 1 if XLEN == 32 else 2
    ok = mxl == expected_mxl and i_ext == 1
    return ok, f"MISA=0x{misa:08x} (MXL={mxl}, I={i_ext}, expected MXL={expected_mxl})"


def t_gpr_roundtrip(sock):
    value = 0xCAFEF00D
    abstract_reg_write(sock, REG_GPR_BASE + 5, value)  # x5
    readback = abstract_reg_read(sock, REG_GPR_BASE + 5)
    ok = readback == value
    return ok, f"x5: wrote 0x{value:08x}, read 0x{readback:08x}"


def t_mem_roundtrip(sock):
    value = 0x13572468
    abstract_mem_write32(sock, SCRATCH_ADDR, value)
    readback = abstract_mem_read32(sock, SCRATCH_ADDR)
    ok = readback == value
    return ok, f"mem[0x{SCRATCH_ADDR:08x}]: wrote 0x{value:08x}, read 0x{readback:08x}"


def t_resume_to_completion(sock):
    # Set resumereq=1 and dmactive=1 in DMCONTROL.
    # DMCONTROL layout: dmactive=bit0, resumereq=bit30, haltreq=bit31
    dmi_write(sock, DM_DMCONTROL, (1 << 30) | 1)
    # Poll DMSTATUS.allhalted until the emulator naturally completes.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        status = dmi_read(sock, DM_DMSTATUS)
        if (status >> 9) & 1:
            return True, f"program halted after resume; DMSTATUS=0x{status:08x}"
        time.sleep(0.02)
    return False, f"timeout; last DMSTATUS=0x{status:08x}"


TESTS = [
    ("idcode",              t_idcode),
    ("dmstatus_initial",    t_dmstatus_initial),
    ("abstract_read_pc",    t_abstract_read_pc),
    ("abstract_read_misa",  t_abstract_read_misa),
    ("gpr_roundtrip",       t_gpr_roundtrip),
    ("mem_roundtrip",       t_mem_roundtrip),
    ("resume_to_completion", t_resume_to_completion),
]


def wait_for_banner(proc, timeout=10.0):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        sys.stdout.write("[simx] " + line.decode(errors="replace"))
        if b"Listening for remote bitbang" in line:
            return True
    return False


def main() -> int:
    proc = subprocess.Popen(
        [SIMX, "-d", "-p", str(RBB_PORT), PROGRAM],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if not wait_for_banner(proc):
        proc.kill()
        print("FAILED: simx did not announce RBB server")
        return 1

    passed = 0
    failed = 0
    try:
        with socket.create_connection(("127.0.0.1", RBB_PORT), timeout=15.0) as s:
            s.settimeout(15.0)
            for name, fn in TESTS:
                try:
                    ok, info = fn(s)
                except Exception as e:
                    ok, info = False, f"exception: {e}"
                status = "PASS" if ok else "FAIL"
                print(f"[{status}] {name}: {info}")
                if ok:
                    passed += 1
                else:
                    failed += 1
            try:
                s.sendall(b'Q')
            except OSError:
                pass
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n{passed}/{passed + failed} DTM tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
