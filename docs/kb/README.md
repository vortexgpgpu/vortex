# Knowledge Base — V80 / Xilinx FPGA Platform

Reference material assembled for the Vortex-on-V80 bring-up. Each document
combines vendor documentation with **[field]**-marked findings measured on this
machine (`orcas2`) that are not documented upstream.

| # | Document | Length | Covers |
|---|---|---|---|
| 1 | [Xilinx / AMD FPGA Driver Development](01_xilinx_fpga_driver_development.md) | ~11 pages | PCIe substrate, Linux PCI/DMA/IRQ APIs, userspace ABI patterns, XDMA vs QDMA vs CPM5, XRT / AVED-AMI / SLASH compared, Linux FPGA subsystem, reconfiguration and reset, AER, debug toolbox, checklist |
| 2 | [Alveo V80 FPGA Configuration](02_v80_fpga_configuration.md) | ~7 pages | XCV80 board spec, PMC/BootROM/PLM/PDI, segmented configuration and the POST race, boot modes, OSPI + FPT layout, multiboot/fallback, CIPS/CPM5 and NoC address map, the four configuration paths, enumeration failures and recovery, bring-up checklist |
| 3 | [SLASH Software and Hardware Architecture](03_slash_architecture.md) | ~9 pages | Layer stack, three-PF topology, kernel module internals, kernel ioctl ABI, vrtd wire protocol, VRT API, vrtbin/`system_map.xml`, platform modes, memory model and buddy allocator, static shell + slashkit connectivity language, v80-smi, testing |

## The findings that cost the most to learn

Cross-referenced here so they are findable without reading all three:

- **`HOST` mastering does not work on the V80 compute shell.** An HLS `m_axi`
  port routed to the QDMA slave bridge yields a master whose reads never
  complete; the identical kernel routed to `HBM0` completes in <0.1 s. Neither
  emulation nor simulation can catch this. — KB-3 §11.4, KB-1 §8.3
- **A PDI design write succeeds only on a freshly reset device.** vrtd resets
  only when the requested shell differs from the current one, so the second load
  in a session always fails and drives the AMC to `NO_AMC`. Use
  `vrt::Device(bdf, vbin, /*program=*/false)`. — KB-2 §9.1
- **If root port `0000:00:01.1` is absent, no PCIe rescan can ever find the
  card.** Firmware does not instantiate a root port whose link is down at POST.
  — KB-2 §8.1, KB-1 §2.4
- **`TOGGLE_SBR` can hard-reset this host**, intermittently and without writing
  an MCE. The fix is a BIOS PCIe fatal-error severity setting, plus unbinding
  every driver before any link transition. — KB-2 §9.3, KB-1 §11
- **The device allocator's minimum block is 4096 bytes.** — KB-3 §10.2
- **`vrt::Kernel::wait()` never times out**; poll `AP_CTRL` yourself. And
  `std::cout` is fully buffered under `runuser`/`tee` — print progress to
  `std::cerr`. — KB-3 §7.3
