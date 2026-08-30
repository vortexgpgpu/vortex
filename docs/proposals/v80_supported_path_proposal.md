# Proposal: abandon the JTAG workarounds and complete the supported install

**Revision 2 — 2026-08-20.** Revision 1's thesis survives review. Four of its
concrete claims did not; see §0. Revision 1 replaced the approach taken in
`v80_hw_bringup_proposal.md`, whose *findings* stand but whose *method* was
wrong.

---

## 0. Revision 2 errata — what review of revision 1 found

Revision 1 was written without checking the board state or running the
commands it prescribed. Four defects, all verified:

**E1. The Phase 2 command cannot work.** §4 prescribed
`sudo v80-smi write-static-shell --flash -d 01:00`. With no `--pdi`, that
defaults to `--shell-type all` and resolves the PDI through
`python3 -m slashkit static-shell-path`. On this install that fails twice
over: slashkit's resolver accepts only `{service,compute}`, so `all` is an
argparse error; and `--shell-type service` resolves
`slashkit.resources.static_shell`, which is an **empty directory** in this
tree — only `static_shell_compute` is populated. The sole working form pins
both axes:

```bash
v80-smi write-static-shell --flash -d <BDF> --shell-type compute --pdi <path>
```

**E2. Phase 1's control run is statistically worthless.** It asked for a
30–60 minute idle run to decide whether the host faults on its own. Measured
baseline MTBF with the card off the bus is ~68 h, so a 45-minute clean run has
a ~1.3% chance of showing a crash *even if the host is definitively faulty*. It
would have burned a reboot and an hour to produce an uninterpretable result.
Dropped.

**E3. "There is no control group" was wrong.** Revision 1 claimed the crash
data could not distinguish a host fault from our operations. A 33-boot
retrospective does distinguish them:

| condition | boots | crashes | uptime | MTBF |
|---|---|---|---|---|
| V80 **on** the PCIe bus | 21 | 18 | 317 h | **1 per 17.6 h** |
| V80 **off** the bus | 12 | 2 | 135 h | **1 per 68 h** |

Both effects are real: the host crashes unaided (boot −19 ran 102 h with the
card off the bus, no drivers, and died), *and* card presence raises the rate
about fourfold. Neither "the machine is faulty" nor "your operations cause it"
is the whole story, and revision 1's flat retraction over-corrected exactly as
the `orcas2-cpu-mce` memory had before it.

**E4. Phase 2 had an unstated prerequisite.** It needs `-d <BDF>`, i.e. the
card enumerated and known to VRTD. Phase 1 as written guaranteed the opposite.
The real sequence needs a JTAG bootstrap and a reboot *first* — which is the
sanctioned use of JTAG under revision 1's own Rule 1, and empirically the
safest observed state (4/4 clean boots).

One claim was **strengthened** by review: the vbin declares
`<ShellType>compute</ShellType>`, and `vrtbin-format.rst` confirms VRT passes
this to vrtd "so the daemon can reset the board to the matching boot
partition". A board reporting `unknown` therefore mismatches on *every*
program. That is the SBR, exactly as §2 claims.

**Thesis:** every recurring problem on this board — unreliable enumeration,
`Shell: unknown`, the forced shell switch, the secondary bus reset, and the
JTAG dependency — traces to a single omission. **The one-time permanent flash
write was never performed.** Everything since has been a workaround for its
absence.

---

## 1. What the documentation actually prescribes

`SLASH-compute/docs/howto/install-from-packages.rst` describes board setup as
**two steps**, not one.

### Step A — temporary bring-up over JTAG

```bash
sudo v80-smi write-static-shell --jtag --no-remove-device
```

Purpose: get *something* into the fabric so the card appears on PCIe. Volatile.

### Step B — permanent image, over PCIe

```bash
sudo v80-smi write-static-shell --flash -d <BDF>
```

> *"The command resolves the packaged static shell PDI, programs it through
> VRTD, and resets the board into the programmed partition. **No host reboot or
> system restart is required.**"*

This writes the SLASH static shell into **OSPI flash**, where it survives power
cycles and is what the card boots from at every subsequent POST.

**We have performed Step A many times. We have never performed Step B.**

---

## 2. How that single omission produced every symptom

| Symptom we fought | Actual cause |
|---|---|
| `v80-smi list` reports `Shell: unknown` | Flash holds an AVED-era image, not the SLASH compute shell. The AMC cannot report a shell it was never given. |
| Every design write forces a shell switch | vrtd compares requested (`compute`) with current (`unknown`). They never match, so `reset_with_ami` runs **every time**. |
| Secondary bus reset on every program | That is what `reset_with_ami` does. It is not a hazard we stumbled into; it is the documented consequence of a shell mismatch we created. |
| Enumeration fails after most reboots | The card boots the flash image, which is not built to release CPM inside the ~120 ms link-training window. A full PDI boot takes ~13.5 s and loses the race. |
| JTAG needed after every reboot | Because the flash image never became correct, the fabric had to be re-populated by hand each time. |
| `PDI load succeeds only on a freshly reset device` | A corollary of the shell mismatch: the only reset available was the one the mismatch triggered. |

One missing command, six derived problems, and roughly two weeks of workarounds
built on top of them.

---

## 3. Where the method went wrong

The findings in the previous proposal were real and are unaffected: the 16-byte
AXI-Lite block rule, the broken `HOST` slave-bridge path, the use-after-free in
the batch submit path. Those are genuine and fixed.

The method was wrong in a specific, nameable way: **I promoted a debug facility
to a production path.**

- `versal_flash_pdi.tcl` issues **`rst` on the PMC target** — a Versal reset. I
  ran it repeatedly, sometimes on a card that was live on the PCIe bus.
- `jtag_load_vortex.sh` performs **partial reconfiguration of a live card**. I
  asserted that PR does not drop the PCIe link. **I never verified that claim.**
- Neither is part of any documented flow. Both are bring-up/debug tools.

I adopted JTAG specifically to avoid the SBR. The SBR existed only because of
the missing flash write. So the workaround was introduced to dodge a problem
that should have been fixed at its source, and it plausibly introduced a worse
one.

### On the host crashes — what is and is not established

Thirteen fatal Bank-5 cache-parity events across seven threads, all recorded by
firmware in BERT.

I claimed these were independent of this project. **That claim is not
supported.** The boot I cited as "nothing of this project running" had no PCIe
drivers but *did* have xsdb/hw_server activity — my own JTAG work. Reviewing
all recent boots:

```
boot | ended | JTAG activity in that boot
 -7  CRASH   YES     -5  CRASH  YES     -3  CRASH  YES     -1  CRASH  YES
 -6  CLEAN   YES     -4  CLEAN  YES     -2  CLEAN  YES
```

Every boot has JTAG activity. **There is no control group**, so the data cannot
distinguish "the machine is faulty" from "these operations cause it". Both
remain live, and the plan below is designed to be safe under either.

Note also that BERT's "cache error" text is *firmware's* classification in a
generic section — the same firmware that chose to reset the box. It is not
proof of a silicon cache defect.

---

## 4. The new approach

**Rule: use only documented flows. JTAG is for bootstrapping a card that cannot
enumerate — nothing else.**

Standing constraints:

1. **No JTAG while the card is on the PCIe bus.**
2. **No partial reconfiguration of a live card.**
3. **No PMC resets.**
4. Programming happens through `v80-smi` / VRT, or not at all.

### The plan (revised — see E1, E2, E4)

**Phase 0 — bootstrap the card onto the bus** *(done 2026-08-20 11:45)*

```bash
bash ~/dev/v80/jtag_load_shell.sh    # no sudo; card is OFF the bus
```

Sanctioned under Rule 1 and empirically the safest observed state. Verified by
the JTAG target hierarchy: `Cortex-R5 #0 (Running)` = AMC firmware up,
`MicroBlaze PPU (Sleeping)` = PLM finished. Then one reboot — the root port
`0000:00:01.1` is created by firmware at POST and no rescan can conjure it.
Historically 4/4 reliable: every JTAG-load boot was followed by a boot with the
card enumerated.

~~Phase 1 — control run~~ **dropped, see E2.**

**Phase 2 — complete the install (the step never done)**

```bash
sudo bash ~/dev/v80/step1_load.sh    # ami + slash + vrtd; reports Shell:
sudo bash ~/dev/v80/step2_flash.sh   # THE MISSING STEP
```

`ami` is required — the flash write goes through the AMC over GCQ
(`GCQ_SUBMIT_CMD_DOWNLOAD_PDI`, 40-minute driver timeout). This is the one
operation for which PF0 genuinely matters.

`step2_flash.sh` issues the E1-corrected command, derives the BDF from
`lspci -d 10ee:50b4 -D` rather than by regexing `v80-smi list` (that output is
free-form; a `hh:mm` clock time matches a BDF pattern), and verifies the FPT
magic `0x92F7A516` on the PDI so a no-FPT JTAG image can never be handed to
`--flash`.

Per the docs it then "resets the board into the programmed partition", with no
host reboot required.

**Phase 3 — verify the omission is closed**

```bash
v80-smi list        # Shell: MUST now read `compute`, not `unknown`
sudo reboot         # a cold-start check
v80-smi list        # must STILL read `compute`, and enumerate without JTAG
```

This is the acceptance test for the whole proposal. If `Shell: compute`
survives a reboot and the card enumerates unaided, every derived problem in §2
is closed at once.

**Phase 4 — run the ladder on the supported path**

With flash holding the compute shell, a design write finds requested == current,
so `reset_with_ami` does not run and **no SBR occurs** — not because we dodged
it, but because there is no longer a mismatch to resolve.

```bash
bash ~/dev/v80/hw_ladder.sh     # programs once, then VORTEX_AVED_NO_PROGRAM=1
```

Rungs: `minimal -l` → `minimal` → `demo` → `sgemv` → `sgemm`, cross-checked
against `instrs=336912`.

---

## 5. Risks

**Flash write failure.** Step B writes OSPI. A failure mid-write can leave the
card unbootable. Mitigations, in order of strength: the JTAG bootstrap does not
read flash at all, so it recovers a card that cannot boot from OSPI — exercised
repeatedly, including today. Additionally, `--shell-type compute` writes boot
**partition 1 only**, leaving partition 0 intact; the documented default of
`all` would have overwritten both and destroyed that fallback. E1 forces us off
the default anyway, which happens to be the safer choice.

**Host instability.** Per E3 the host does fault on its own (~1 per 68 h) and
card presence raises it to ~1 per 17.6 h. A flash write is therefore exposed:
at that rate a 40-minute window carries roughly a 4% chance of a reset landing
inside it. That is not negligible, but it is the *tolerable* kind of risk,
because the failure is recoverable via JTAG rather than terminal. Waiting for
stability is not an option — there is no fix for the host fault within this
project's scope, and E2 shows we cannot even measure it in reasonable time.
Mitigation is to keep the exposed window short and to touch nothing else on the
machine while it runs.

**`ami` must be loaded for Phase 2.** That contradicts the "slash-only"
mitigation from yesterday. It is unavoidable: the flash path runs through the
AMC. It is also one-time.

---

## 6. What is already complete

Unchanged by any of this. Eleven commits, nothing pushed.

| | |
|---|---|
| AXI-Lite 16-byte block rule | fixed in RTL, verified on silicon, lint gates synthesis |
| `HOST` slave bridge unusable | worked around via HBM1 staging; `minimal -l` PASSED on silicon |
| Use-after-free in batch submit | found by reading, fixed, validated in simulation |
| Transport gate, poll timeouts, `avedsim` build | done |
| Knowledge bases, upstream defect reports | written |

The software is finished. This proposal is about the *platform state*, which is
where the real defect always was.

---

## 7. Acceptance criteria

1. ~~Phase 1 control run~~ — replaced by the 33-boot retrospective in E3.
2. `v80-smi write-static-shell --flash --shell-type compute --pdi …` completes
   successfully.
3. `v80-smi list` reports `Shell: compute` **and still does after a reboot**.
4. The card enumerates after a reboot with **no JTAG intervention**.
5. A design write completes with no `reset_with_ami` and no SBR in the journal.
6. Ladder rungs 1–5 pass; `sgemm` reports `instrs=336912`.

Criterion 4 is the one that matters most: it is the direct disproof of the
condition that has driven every workaround for two weeks.
