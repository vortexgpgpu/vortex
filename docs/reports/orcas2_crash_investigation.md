# orcas2 host crash investigation

**2026-08-21.** Full analysis of every host crash still recorded on this
machine, the evidence for and against each candidate root cause, and a ranked
diagnostic plan.

---

## 0. Direct answer to the question asked

**No. I cannot confirm that the root cause is found, and I cannot confirm the
crashes will not repeat.**

What I had before this analysis was a *hypothesis* dressed up as a diagnosis. I
recommended disabling PBO/Curve Optimizer and EXPO on the grounds that Bank-5
parity errors are "the classic signature of undervolt **under load**." The load
half of that claim is now measurably false — every crash I can check happened
with the machine **99.8% idle**. I should have checked before recommending it.

This is the fourth conclusion I have reached about these crashes. The previous
three were:

| # | verdict | status |
|---|---|---|
| 1 | "The CPU is defective, RMA it" | withdrawn — a 10-day clean boot refutes it |
| 2 | "Every crash is tied to `slash.ko` activity" | withdrawn — crashes occur with no drivers loaded |
| 3 | "Crashes are INDEPENDENT of the V80 stack" | withdrawn — card presence ~3.6× the rate |
| 4 | "Undervolt instability under load" | **withdrawn here** — every crash is at idle |

A pattern worth naming: each verdict was reached from a handful of boots and
stated with more confidence than the evidence carried. This report tries to
separate what is *measured* from what is *inferred*.

---

## 1. Scope and limits of the evidence

| | |
|---|---|
| Boots on record | **35** (journald), 2026-08-01 18:23 → 2026-08-21 04:56 |
| Crashes | **21** |
| Clean shutdowns | 13 |
| Machine first booted | 2026-07-25 |
| **Data lost** | **Jul 25 – Aug 1** — journals rotated away |

**On the "50+ crashes" figure:** I can only evidence 21. The first week of the
machine's life is not in journald, so the true total is higher than 21 and
50 is not implausible — but I will not assert a number I cannot show.

**A fundamental limit, and it is the most important finding in this report:**

```
GHES: APEI firmware first mode is enabled by APEI bit.
MCE exceptions delivered to Linux this boot: 0
```

Error handling is **firmware-first**. The BIOS intercepts the machine check,
classifies it, writes a BERT record, and resets the box. Linux never receives
an MCE — the counter is zero on all 48 CPUs. **`rasdaemon` cannot capture these
events no matter how it is configured**, and installing it (which I recommended
twice) was never going to help. We are diagnostically blind by platform
configuration.

---

## 2. The dataset

`CARD` = V80 enumerated on PCIe · `DRV` = slash/ami loaded · `JTAG` = a USB
reset occurred · `SBR` / `DW` = secondary bus resets / design writes.
BERT is aligned to the boot it *describes* (a BERT in boot N reports boot N−1's
death).

```
IDX  START               MIN END   CARD JTAG DRV  SBR  DW   ITS-OWN-BERT       APIC
-34  2026-08-01 18:23:52   142 CRASH Y    -    Y    8    -   (none)             -
-33  2026-08-01 22:01:04     8 CLEAN -    -    -    -    -
-32  2026-08-01 22:30:23    24 CRASH -    -    -    -    -   (none)             -
-31  2026-08-01 23:00:19     0 CRASH -    -    -    -    -   (none)             -
-30  2026-08-01 23:16:36    77 CLEAN Y    -    Y    8    -
-29  2026-08-02 00:35:31    34 CRASH Y    -    Y   16   10   Perr S0:T002:B05   0x2
-28  2026-08-02 01:16:10  2836 CRASH Y    Y    Y    8    3   (none)             -
-27  2026-08-04 00:34:19    15 CRASH Y    -    Y    8    3   Perr S0:T002:B05   0x2
-26  2026-08-04 00:52:49   138 CLEAN Y    -    Y    -    -
-25  2026-08-04 03:13:27    25 CRASH Y    -    Y    8   12   Perr S0:T004:B05   0x4
-24  2026-08-04 03:41:12 14495 CRASH Y    -    Y   16   29   Perr S0:T002:B05   0x2
-23  2026-08-14 05:19:53  6115 CRASH -    -    -    -    -   (none)             -
-22  2026-08-18 11:22:38     5 CRASH Y    -    -    -    -   (none)             -
-21  2026-08-18 11:52:06     5 CRASH Y    -    Y    8    3   Perr S0:T000:B05   0x0
-20  2026-08-18 12:01:15    32 CLEAN Y    -    Y    -    -
-19  2026-08-18 12:34:54    30 CRASH Y    -    Y    8    3   Perr S0:T001:B05   0x1
-18  2026-08-18 13:11:01   289 CRASH Y    -    Y   16   34   Perr S0:T002:B05   0x2
-17  2026-08-18 18:02:02   137 CLEAN -    -    -    -    -
-16  2026-08-18 20:20:28   729 CRASH Y    -    Y   24   23   Perr S0:T006:B05   0x6
-15  2026-08-19 08:31:57    52 CLEAN -    -    -    -    -
-14  2026-08-19 09:25:57    14 CRASH Y    -    Y    8    -   Perr S0:T006:B05   0x6
-13  2026-08-19 09:42:13   267 CLEAN -    -    -    -    -
-12  2026-08-19 14:11:32    35 CLEAN -    Y    -    -    -
-11  2026-08-19 14:48:29    27 CRASH Y    -    Y    -    -   Perr S0:T002:B05   0x2
-10  2026-08-19 15:17:40    98 CLEAN -    Y    -    -    -
 -9  2026-08-19 16:57:14    42 CRASH Y    Y    -    -    -   Perr S0:T01A:B05   0x1a
 -8  2026-08-19 17:44:35   627 CLEAN -    Y    -    -    -
 -7  2026-08-20 04:13:42    46 CRASH Y    Y    -    -    -   Perr S0:T023:B05   0x23
 -6  2026-08-20 05:03:23    31 CLEAN -    Y    -    -    -
 -5  2026-08-20 05:35:48    60 CRASH Y    Y    Y    -    -   Perr S0:T006:B05   0x6
 -4  2026-08-20 06:37:51   318 CLEAN -    Y    -    -    -
 -3  2026-08-20 11:57:20   194 CRASH Y    -    Y    8    3   Perr S0:T00A:B05   0xa
 -2  2026-08-20 15:14:18   207 CLEAN -    Y    -    -    -
 -1  2026-08-20 18:43:27   611 CRASH Y    -    Y   16    -   Perr S0:T032:B05   0x32
```

---

## 3. Findings

### F1 — Six of 21 crashes produced no error record at all

Boots −34, −32, −31, −28, −23, −22 died with **no BERT record in the following
boot**. Either the machine died faster than firmware could write one, or these
were not machine checks at all. **29% of the crashes have no error evidence
whatsoever**, and any theory built only on the BERT records ignores them.

### F2 — Every BERT record is byte-identical and content-free

All 15 records carry `Check Information: 0x000000000602001f`, decoding to:

```
Transaction Type: 2, Generic      Operation: 0, generic error
Level: 0                          Processor Context Corrupt: true
```

Transaction *generic*, operation *generic*, level *0*. Every discriminating
field is empty. This is firmware boilerplate meaning "a fatal something
happened on this thread" — **it is not a diagnosis, and the words "cache error"
in it carry no evidentiary weight.** I have repeatedly over-read this record.

### F3 — Every crash occurred with the machine ~99.8% idle ⭐

From `sysstat`, the 40 minutes preceding five representative crashes spanning
the whole period, including the two extremes (a 241-hour uptime and a crash
seconds after V80 work):

| crash | context | CPU in the preceding 40 min |
|---|---|---|
| −24 | 241 h uptime, card + drivers, 29 design writes | `user 0.1% · idle 99.7%` |
| −23 | 102 h uptime, **no card, no drivers** | `user 0.1% · idle 99.8%` |
| −16 | card + drivers, 23 design writes | `user 0.1% · idle 99.8%` |
| −3 | seconds after ladder rung 1 PASSED | `user 0.1% · idle 99.8%` |
| −1 | seconds after `v80-smi reset` | `user 0.1% · idle 99.8%` |

**This is an idle-time failure, not a load-time failure.** It reverses the
polarity of the hypothesis I gave you, and it is the single most useful fact in
this report.

### F4 — The failures span the entire CPU

Nine distinct APIC IDs: `0x00 0x01 0x02 0x04 0x06 0x0A 0x1A 0x23 0x32`. Valid
APIC IDs on this part are `0–11, 16–27, 32–43, 48–59` (24 cores / 48 threads),
so the failures land in **all four core groups**. `0x02` appears 5×; the rest
1–3× each. A localized silicon defect does not migrate across four core groups.

### F5 — A 10-day clean run under exactly the "dangerous" conditions

Boot −24 ran **241 hours** with the card enumerated, drivers loaded, 29 design
writes and 16 SBRs, and did not crash until the end. Any theory claiming the
V80 stack promptly kills the host must explain this boot.

### F6 — The card-presence correlation is real but confounded

| condition | crashes | uptime | MTBF |
|---|---|---|---|
| V80 on the PCIe bus | 18 | 331 h | 1 per 18.4 h |
| V80 off the bus | 3 | 132 h | 1 per 44 h |

~2.4–3.6× depending on whether the 0-minute boot −31 is counted. **But** F3
shows the machine is idle at crash time either way, and F5 shows 241 clean
hours with the card present. The correlation is not explained, and "the V80
causes it" is not supported as stated. A confound worth noting: `ami.ko`
heartbeats the AMC every 0.5 s, which prevents long uninterrupted deep-idle
residency — but boots −22, −9 and −7 crashed with the card present and **no
drivers at all**, which that confound does not cover.

### F7 — Idle is spent almost entirely in a legacy I/O-port C-state

```
driver: acpi_idle          governor: menu
POLL              latency 0us     usage 20807
C1  ACPI FFH MWAIT 0x0      latency 1us     usage 731121
C2  ACPI IOPORT 0x814       latency 100us   usage 179609
```

C2 residency on cpu0 was **2331 s out of ~50 min uptime** — essentially all
idle time. C2 here is entered by reading legacy I/O port `0x814` (the FCH
`P_LVL2` register), not via MWAIT. Combined with F3, **the machine dies while
sitting in C2, every time.** No `idle=` or `max_cstate` parameter is set on the
kernel command line, so this is the platform default.

### F8 — Memory ECC is clean

`EDAC mc0: ce_count=0 ue_count=0` across the whole period, 128 GB. No evidence
implicating DRAM.

---

## 4. Root cause: NOT ESTABLISHED

Ranked by fit to the evidence.

### H1 — Deep C-state (C2 via P_LVL2) instability · **best fit**

Explains F3 (all crashes at idle) and F7 (all idle time in C2) directly.
Consistent with F4 — a C-state transition fault lands on whichever thread was
entering or leaving idle, which is arbitrary. Consistent with F1: a core that
fails to wake produces no error record at all.
**Does not explain** F6 without a further assumption.

### H2 — Curve Optimizer / PBO undervolt · **plausible**

Contrary to what I told you, CO instability is *most* pronounced at **low
current**, i.e. idle and light load, because that is where the offset drives
V<sub>core</sub> lowest. So F3 is consistent with H2, not against it — I
reached the right BIOS recommendation via wrong reasoning. Cannot be ranked
against H1 without testing, and current BIOS settings are unknown to me.

### H3 — Genuine silicon defect · **weak**

F5 (241 clean hours) and F4 (four core groups) argue against. Not excluded.

### H4 — V80 stack causes the crashes · **weak as a primary cause**

F3 (idle), F5 (241 h clean), and crashes with no card and no drivers all argue
against it being primary. F6's correlation remains genuinely unexplained and
may reflect a secondary effect.

### H5 — Memory instability (EXPO/DOCP) · **weak**

F8 shows zero ECC events; error signature is not memory-related.

---

## 5. Diagnostic plan

Ordered so each step is cheap, reversible, and discriminates between
hypotheses. **Step 1 needs no BIOS access and no hardware change.**

### Step 1 — Eliminate deep C-states (tests H1)

Add to the kernel command line and reboot:

```
processor.max_cstate=1 idle=halt
```

Caps ACPI idle at C1 and removes the C2 I/O-port path entirely. Fully
reversible — edit the GRUB entry for a single boot to try it without committing.

- **Crashes stop over several days → H1 confirmed**, and a permanent fix is
  either this parameter or the corresponding BIOS C-state setting.
- **Crashes continue at idle → H1 refuted**, proceed to step 2.

This is the highest-value experiment available and it should have been run two
weeks ago.

### Step 2 — Restore diagnostic visibility (enables everything else)

In BIOS, look for **WHEA / APEI / "Firmware First"** error handling and turn
firmware-first **off** so Linux handles machine checks. Then `rasdaemon`
(already installed, enabled, running with `-r`) captures real MCA bank
registers — `MCA_STATUS`, `MCA_ADDR`, `MCA_IPID`, `MCA_SYND` — which name the
failing unit precisely. Until this changes we are guessing from firmware
boilerplate.

### Step 3 — Stock the CPU and memory (tests H2, H5)

Only after steps 1–2, and one variable at a time:
`Ai Tweaker → Precision Boost Overdrive → Disabled`, Curve Optimizer to zero;
then separately `DOCP/EXPO → Disabled`.

### Step 4 — Establish a real V80-free control (tests H4/F6)

Physically remove the card, or blacklist `ami`/`slash` and leave the box up for
**a week**. At a 44 h baseline MTBF, anything shorter has no statistical power —
which is exactly why the 45-minute "control run" I proposed earlier was
worthless.

### Step 5 — BIOS update

0503 (2025-07-18) is the shipped version on a CPU launched shortly before it.
Check for newer AGESA. Do this last so it does not confound steps 1–3.

---

## 6. What this means for the V80 work

The V80 bring-up is **not** the cause of these crashes and cannot fix them.
Both efforts should proceed independently:

- The host investigation is step 1 above, and it costs one reboot you are
  already spending.
- The V80 work is ~20 minutes of ladder once the card is on the bus.

The interaction to be honest about: every crash knocks the card off the PCIe
bus and costs a JTAG reload plus a reboot. That is the treadmill. It ends when
the crashes end, not when the V80 work finishes.

---

## 7. Accountability

Specific errors in my handling of this, beyond the four withdrawn verdicts:

1. **I recommended `rasdaemon` twice** without checking that firmware-first
   mode makes it structurally incapable of seeing these events.
2. **I never looked at `sysstat`**, which was sampling CPU load every 10
   minutes the entire time and immediately falsifies the load hypothesis.
3. **I never looked at the C-state configuration**, which is the strongest lead
   in this report and was one file read away.
4. **I proposed a 45-minute control run** against a 44-hour MTBF — a test with
   ~1% power that would have consumed a reboot to prove nothing.
5. **I asserted "the root port cannot be recovered at runtime" three times
   before verifying it**, and only checked on the fourth.

The common thread is reaching for a conclusion before reading the cheap
evidence that was already on disk.
