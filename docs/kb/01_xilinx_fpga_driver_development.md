# Xilinx / AMD FPGA Driver Development — Knowledge Base

**Scope.** Everything a systems programmer needs to write, read, or debug a
host-side Linux driver for an AMD (Xilinx) PCIe FPGA accelerator: the PCIe
substrate, the Linux PCI/DMA/IRQ APIs, the userspace ABI patterns, the AMD DMA
IP family (XDMA, QDMA, CPM4/CPM5), the three competing runtime/driver stacks
(XRT, AVED/AMI, SLASH), the Linux FPGA subsystem, and — the part that costs the
most time in practice — reconfiguration, reset, and error handling.

Written against Linux 6.x, Vivado/Vitis 2025.1, and the AMD Alveo V80. Sections
marked **[field]** are empirical findings from the Vortex-on-V80 bring-up on
this machine, not vendor documentation.

---

## 1. The mental model

An FPGA accelerator card is not one device. It is a *set of PCIe functions* that
happen to share a die, plus a *reconfigurable fabric* whose contents can change
under the driver's feet, plus (on Versal) a *management processor* running its
own firmware that the host talks to over a mailbox.

Three consequences follow, and nearly every hard bug traces back to one of them:

1. **The device identity is not stable.** Reprogramming the fabric can change
   BAR sizes, device IDs, and which functions exist. A driver that caches
   anything across a reconfiguration is wrong.
2. **The link can go down while the host is still driving it.** Any PCIe reset,
   partial reconfiguration, or PDI reload drops the link. If a driver is still
   bound and touches MMIO or config space during that window, the CPU gets an
   unrecoverable bus error. On some platforms that is a machine check and the
   host resets. **[field]**
3. **Two masters share the bus.** The host masters into the card (MMIO, DMA
   reads/writes), and the card masters into the host (DMA, or an AXI master
   pointed at host DRAM through a slave bridge). Direction-specific bugs are
   common and each direction has an independent failure mode.

Keep those three in mind and the rest of this document is detail.

---

## 2. PCIe substrate

### 2.1 Enumeration and configuration space

At power-on, platform firmware walks the PCIe tree, assigns bus numbers, sizes
and assigns BARs, and hands Linux a topology. Linux re-walks it and can
re-assign, but **it cannot create a root port that firmware did not
instantiate**.

> **[field] The single most expensive fact in this project.** AMD platform
> firmware does not instantiate a root port whose downstream link is not up at
> POST. On this host the V80 lives behind `0000:00:01.1`. If the card is not
> link-trained when firmware enumerates, `00:01.1` never appears in
> `/sys/bus/pci/devices/` at all, the bus numbers shift so that a different
> device (`00:07.1`, AMD `1022:1556`) takes bus 01, and **no amount of
> `echo 1 > /sys/bus/pci/rescan` can ever find the card** — there is no bridge
> to rescan behind. The only fixes are (a) make the card win the link-training
> race at POST, or (b) cold power-cycle. See §2.4 and KB-2 §8.

Config space layout:

| Region | Offsets | Contents |
|---|---|---|
| Type 0 header | `0x00`–`0x3F` | Vendor/Device ID, Command, Status, BAR0–5, Subsystem IDs, Interrupt Line/Pin |
| Capability list | `0x40`–`0xFF` | MSI, MSI-X, PCIe Capability, Power Management, Vendor-Specific |
| Extended config | `0x100`–`0xFFF` | AER, ARI, SR-IOV, Vendor-Specific Extended (VSEC), DVSEC |

AMD designs use **VSEC in extended config space to publish hardware discovery
metadata** — on AVED, the addresses of the UUID ROM, the mailbox, and the
address remapper are published there so the driver can find its peripherals
without a hardcoded map. This is the right pattern to copy: put a small,
version-tagged discovery table in extended config space and let the driver walk
it. It survives fabric changes that a hardcoded offset table does not.

### 2.2 BARs

A BAR is a window from host physical address space into the device. Key
properties a driver must respect:

- **Type**: memory (`IORESOURCE_MEM`) vs I/O port. Always use memory BARs.
- **Width**: 32-bit or 64-bit. A 64-bit BAR consumes two BAR slots.
- **Prefetchable**: allows the host to combine/prefetch. Never mark a register
  BAR prefetchable — writes may be coalesced and reads may be speculative.
- **Size**: fixed by the hardware at synthesis time. If a reprogram changes it,
  the kernel must re-size the resource, which usually means remove + rescan.

Typical Alveo/Versal usage: a small BAR for control registers, a large BAR
windowing card DRAM/HBM, and a separate function's BAR for the DMA engine's own
registers.

### 2.3 Physical functions

AMD accelerator designs split responsibilities across PFs so that management
survives when the user function is wedged, and so that a hypervisor can pass
through the user function while keeping management in the host.

| Stack | Management PF | User/DMA PF | Control PF |
|---|---|---|---|
| XRT / Alveo U-series | `xclmgmt` | `xocl` | — |
| AVED (V80 reference) | PF0 `ami` (`10EE:50B4`) | PF1 QDMA (`10EE:50B5`/`50BD`) | — |
| SLASH (V80 compute) | PF0 `ami` (`10EE:50B4`) | PF1 `slash_qdma` (`10EE:50C1`) | PF2 `slash_ctl` (`10EE:50C2`) |

The BDF convention throughout: a *board BDF* is the `DDDD:BB:DD` prefix and each
function is `.0` / `.1` / `.2`. Tools accept the prefix and derive the rest.

### 2.4 Link training and the POST race **[field]**

Versal devices boot from OSPI in roughly 13.5 s on the V80. PCIe requires the
endpoint to be ready far sooner than that. AMD solves this with **segmented
configuration**: the PDI is structured like a Tandem PROM image so that CPM (the
PCIe hard block) and its minimum support logic are configured and released
first, meeting the ~120 ms link-training window, with the rest of the design
loaded afterwards.

Practical consequences:

- A design that is *not* built with segmented configuration will lose the POST
  race and the card will not enumerate after a warm reboot.
- A configuration loaded over **JTAG survives PERST#**, because it is already
  resident when the reset arrives. An OSPI boot restarts on PERST and loses the
  race. This is why "JTAG-load the shell, then reboot" recovers a card that a
  plain reboot cannot. **[field]**
- The corollary trap: a JTAG shell load leaves the management firmware reporting
  `Shell: unknown`, which forces a *shell switch* on the next design load, which
  runs a secondary bus reset — see §12.3.

---

## 3. The Linux PCI driver skeleton

```c
static const struct pci_device_id my_ids[] = {
    { PCI_DEVICE(0x10EE, 0x50C2) },
    { PCI_DEVICE(0x10EE, 0x50B6) },   /* legacy ID, same role */
    { 0, }
};
MODULE_DEVICE_TABLE(pci, my_ids);

static struct pci_driver my_driver = {
    .name     = "slash_ctl",
    .id_table = my_ids,
    .probe    = my_probe,
    .remove   = my_remove,
};
module_pci_driver(my_driver);
```

`MODULE_DEVICE_TABLE` is what lets udev/modprobe autoload the module when the
device appears. Omit it and the driver only binds on manual `insmod` + rescan.

### 3.1 probe()

The canonical order, and why each step exists:

```c
static int my_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    int err;

    /* 1. Reject functions this driver does not own. Guards against
     *    device-ID collisions and misconfigured bitstreams. */
    if (PCI_FUNC(pdev->devfn) != EXPECTED_PF)
        return -EINVAL;

    /* 2. Take a reference if anything will outlive probe(). */
    pci_dev_get(pdev);

    /* 3. Power the function up and enable memory decoding. */
    err = pci_enable_device(pdev);
    if (err) goto err_put;

    /* 4. Required before the device can master DMA. Forgetting this
     *    produces a device that silently never writes to host memory. */
    pci_set_master(pdev);

    /* 5. Claim the BAR regions so nothing else maps them. */
    err = pci_request_regions(pdev, DRV_NAME);
    if (err) goto err_disable;

    /* 6. Map the register BAR. */
    priv->mmio = pci_iomap(pdev, 0, 0);   /* 0 = whole BAR */
    if (!priv->mmio) { err = -ENOMEM; goto err_release; }

    /* 7. Declare DMA addressing capability BEFORE any mapping. */
    err = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
    if (err) err = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
    if (err) goto err_unmap;

    /* 8. Interrupts. */
    err = pci_alloc_irq_vectors(pdev, 1, nvec, PCI_IRQ_MSIX | PCI_IRQ_MSI);
    ...
    /* 9. Only now expose anything to userspace. */
    err = my_cdev_create(pdev);
    ...
}
```

**Rule: create the userspace-visible node last, destroy it first.** Any other
order gives userspace a window to issue an ioctl against half-initialised state.
`slash_pcie.c` follows this exactly — `slash_ctldev_create()` is the final step
of probe and `slash_ctldev_destroy()` the first step of remove.

### 3.2 remove()

Strict reverse order of probe. `remove()` is called both on `rmmod` *and* when
the device disappears from the bus (surprise removal, or a deliberate
`pci_stop_and_remove_bus_device()`). It must therefore be safe to call when the
device is already gone — MMIO reads will return `0xFFFFFFFF` and writes will be
discarded. Never poll for a hardware acknowledgement in `remove()`.

---

## 4. MMIO discipline

Use the accessors, never a raw dereference:

```c
u32 v = readl(priv->mmio + REG_STATUS);
writel(val, priv->mmio + REG_CTRL);
```

Points that bite:

- **Writes are posted.** `writel()` returns before the device has seen the
  write. To order a write against a subsequent read of a *different* device,
  read back a register from the same BAR — a read is non-posted and flushes the
  write ahead of it.
- **`readl()` after link-down returns `~0`.** `0xFFFFFFFF` is the universal "the
  device is gone or the completion timed out" value. Any polling loop on a
  status register must treat `~0` as a fatal condition, not as a set of status
  bits. A loop that waits for `bit1 == 1` will happily "succeed" against a dead
  device, because `~0` has bit 1 set. **[field]** — this class of bug wasted
  hours here.
- **`memcpy_toio()` / `memcpy_fromio()`** for bulk BAR access; plain `memcpy()`
  on an `ioremap`ed pointer is undefined.
- **Ordering vs. DMA.** `writel()` is ordered with respect to prior writes to
  the same device, but not with respect to DMA-visible writes to host memory.
  Before ringing a doorbell that makes the device read a descriptor you just
  wrote to coherent memory, issue `wmb()`. Missing this produces the classic
  "the device fetched a stale descriptor" bug.
- **Userspace mappings need explicit bracketing.** When a BAR is exported to
  userspace as a dma-buf (§7.3), userspace has no `readl()`. SLASH requires
  `DMA_BUF_IOCTL_SYNC` with `DMA_BUF_SYNC_START|_READ` / `_WRITE` around every
  access to get the ordering guarantees. Design your ABI so this is enforced by
  a helper, not by documentation.

---

## 5. The DMA API

### 5.1 Coherent vs streaming

**Coherent (consistent)** — allocated once, mapped for the device's lifetime,
CPU and device see the same data without explicit sync. Use for descriptor
rings, doorbell/completion structures, and anything both sides poll.

```c
void *cpu = dma_alloc_coherent(&pdev->dev, len, &dma_handle, GFP_KERNEL);
/* cpu       — kernel virtual address
 * dma_handle— bus address the DEVICE uses; NOT a physical address, and
 *             NOT a CPU virtual address. Under an IOMMU it is an IOVA. */
dma_free_coherent(&pdev->dev, len, cpu, dma_handle);
```

This is exactly what `slash_hostbuf.c` does: it allocates coherent host memory,
hands userspace an `mmap`able view of `cpu_addr`, and hands the *device* the
`dma_addr`. That combination — one buffer, two address views — is what lets an
FPGA kernel keep a command ring in host DRAM and pull from it with no host-side
sync. It is the cleanest pattern for host-mastered rings and worth knowing.

**Streaming** — map an existing buffer for one transfer, unmap after.

```c
dma_addr_t d = dma_map_single(&pdev->dev, buf, len, DMA_TO_DEVICE);
if (dma_mapping_error(&pdev->dev, d)) { ... }
/* ... device transfers ... */
dma_unmap_single(&pdev->dev, d, len, DMA_TO_DEVICE);
```

For user pages: pin with `pin_user_pages_fast()`, build a `struct
scatterlist`/`sg_table`, then `dma_map_sgtable()`. The QDMA driver's fast path is
exactly this, and the SLASH driver avoids it entirely by owning the pages itself
(`BUF_CREATE` allocates, builds the SGL, and DMA-maps once, so the per-transfer
cost is zero).

### 5.2 DMA masks

Set the mask **before** any mapping call. It tells the DMA layer (and the IOMMU)
what address range the device can generate. Getting this wrong on a 64-bit
capable device silently forces bounce buffers through the 32-bit SWIOTLB pool
and destroys throughput.

### 5.3 IOMMU

With an IOMMU enabled (AMD-Vi / Intel VT-d), `dma_addr_t` is an IOVA, not a
physical address. Two implications:

- A device address obtained from `dma_alloc_coherent()` is only valid for *that
  device*. Handing it to a second device (or to a kernel that assumes physical
  addressing) is a bug. `slash_hostbuf.c` refuses dma-buf attachments for
  exactly this reason.
- `iommu=pt` (passthrough) makes IOVA == physical for that device, which is
  faster and simpler for a trusted accelerator, at the cost of losing the
  IOMMU's protection. Many FPGA deployments run this way.

---

## 6. Interrupts

```c
int nvec = pci_alloc_irq_vectors(pdev, 1, want, PCI_IRQ_MSIX | PCI_IRQ_MSI | PCI_IRQ_INTX);
if (nvec < 0) return nvec;
for (i = 0; i < nvec; i++)
    request_irq(pci_irq_vector(pdev, i), my_isr, 0, name, ctx);
```

- **MSI-X** is what you want: up to 2048 individually-routable vectors, each
  with its own address/data pair, so per-queue interrupts are possible. AVED's
  V80 CIPS configuration allocates **8 MSI-X vectors on PF1**.
- **MSI** requires a power-of-two contiguous block and is limited to 32.
- **INTx** is a shared level-triggered line; the ISR must check "was it me?" and
  return `IRQ_NONE` otherwise.
- MSI/MSI-X are delivered as posted memory writes to the LAPIC region, so they
  are ordered behind the device's prior DMA writes. That is why a completion
  interrupt implies the data is visible — but only if the device issued the DMA
  writes before the MSI, which is an IP guarantee you should verify rather than
  assume.

**Polling is a legitimate design.** QDMA supports a poll mode and SLASH's
runtime paths largely poll. For an accelerator with microsecond-scale kernels,
interrupt latency dominates and polling wins. Provide both; make it a module
parameter (`qdma_num_threads` in SLASH is the analogous knob).

---

## 7. Userspace ABI design

This is where most FPGA drivers age badly. The fabric changes every synthesis
run; the ABI must not.

### 7.1 Character device or sysfs?

The AMI driver's stated policy is the right default:

> "Where possible, device attributes are exposed via the sysfs subsystem… Use of
> ioctl within AMI is restricted to 'edge cases' and is used by exception, for
> example, in dealing with binary data."

Sysfs for scalars and enumerable state (sensors, versions, identity, debug
verbosity via `/sys/bus/pci/drivers/ami/ami_debug_enabled`). ioctl for anything
with a binary payload, a file descriptor result, or a multi-field atomic
operation.

### 7.2 Versioned ioctl structs

The pattern SLASH uses, and the one to copy:

```c
struct slash_ioctl_bar_info {
    __u32 size;           /* [in/out] caller sets sizeof(struct) */
    __u8  bar_number;     /* [in]  */
    __u8  usable;         /* [out] */
    ...
    __u64 start_address;  /* [out] */
    __u64 length;         /* [out] */
};
```

The kernel reads `size` first, copies in `min(user_size, kernel_size)` bytes,
zero-fills fields the caller's older struct lacks, writes back
`min(user_size, kernel_size)`, and `clear_user()`s any tail when the caller's
struct is *newer*. Result: the driver and the library version independently, in
both directions, with no ioctl-number churn. Combine with:

- Fixed-width `__u32`/`__u64` types only. No `long`, no `size_t`, no enums
  without an explicit underlying width.
- Explicit padding fields, always zeroed. Reject non-zero reserved fields so
  they can gain meaning later.
- One magic byte per device class, sequential command numbers
  (SLASH: magic `'v'` = 0x76, control device `0x30`–`0x32`).
- `-ENOTTY` for unknown commands. Never `-EINVAL` — `ENOTTY` is how userspace
  probes for feature support.

### 7.3 Returning file descriptors

Two idioms, both used by SLASH:

- **dma-buf for BAR mappings.** `GET_BAR_FD` returns a dma-buf fd *as the ioctl
  return value* (not as a struct field), which userspace `mmap()`s. This gets
  you refcounted lifetime, cross-process sharing, and the `DMA_BUF_IOCTL_SYNC`
  bracketing hook for free, instead of hand-rolling an `mmap` file operation.
- **anon-inode fds for channels.** `QPAIR_GET_FD` returns a per-queue-pair fd
  whose `file_operations` carry the actual I/O path. This makes the queue a
  first-class kernel object with its own lifetime, and lets you attach
  `io_uring_cmd` support (SLASH does — see `slash_qdma_qpair_uring_cmd`) for
  async submission without a thread per transfer.

Note explicitly in your documentation that **BAR mappings are not inherited
across `fork()`** — each child must obtain its own fd.

### 7.4 Naming and identity

SLASH creates `/dev/slash_ctl<N>` with a `miscdevice`, plus a sysfs alias
`/sys/class/misc/slash_ctl_<BDF>/`. The critical documented caveat:

> "The mapping of one file path to a physical card is not guaranteed across
> remove+rescan cycles and userspace should always verify the BDF identity of
> the accessed card."

And worse, the `<N>` for `slash_ctl` and `slash_qdma_ctl` are allocated from
*separate* counters, so `slash_ctl0` and `slash_qdma_ctl1` can be the same
board. Any driver that hotplugs must provide a BDF-query ioctl
(`GET_DEVICE_INFO`) and userspace must use it. Do not index by node number.

---

## 8. The AMD DMA IP family

| IP | Doc | Typical device | Model |
|---|---|---|---|
| XDMA (DMA/Bridge Subsystem for PCIe) | PG195 | UltraScale+ Alveo, 7-series | Fixed small number of H2C/C2H channels, scatter-gather descriptor chains |
| QDMA (Queue DMA Subsystem for PCIe) | PG302 | Alveo U-series, Versal soft IP | Thousands of queues, per-queue contexts, MM and Stream |
| CPM4 / CPM5 QDMA | PG347 | Versal Prime/Premium/HBM hard block | QDMA semantics in hardened silicon inside CPM |

Both drivers live in one repository: `github.com/Xilinx/dma_ip_drivers`, with
`XDMA/linux-kernel/` and `QDMA/linux-kernel/` trees.

### 8.1 XDMA

Structure: `libxdma.c` (engine core) + `xdma_mod.c` (PCI registration, char
device instantiation) + `cdev_sgdma.c` (the SG DMA char devices). Userspace sees
`/dev/xdma<N>_h2c_<ch>`, `/dev/xdma<N>_c2h_<ch>`, `/dev/xdma<N>_control`,
`/dev/xdma<N>_user`, and — a genuinely useful feature — `/dev/xdma<N>_xvc`, a
**Xilinx Virtual Cable** node that carries JTAG over PCIe so Vivado Hardware
Manager can debug the fabric with no physical JTAG cable.

XDMA's model is simple: a small fixed set of channels, each a descriptor engine.
Read/write on the char device does a blocking SG transfer. It is the right
choice when you have a handful of high-bandwidth streams and no need for
per-tenant isolation.

### 8.2 QDMA

QDMA replaces "a few channels" with "a queue namespace". Three layers:

1. **`qdma-pf.ko` / `qdma-vf.ko`** — PCI probe, char devices, sysfs.
2. **`libqdma`** — the engine: descriptor fetch, queue state machine, memory
   mapping. Key structures:
   - `struct xlnx_dma_dev` — the device: BAR mappings, capabilities, config.
   - `struct qdma_descq` — one queue: descriptor ring, completion status,
     producer/consumer indices, state.
3. **Hardware Access Layer** — per-IP-version register access, selected by
   runtime IP detection:

   | IP version | HAL source | Max queues |
   |---|---|---|
   | CPM5 (Versal hard) | `eqdma_cpm5_access.c` | 4095 |
   | Soft IP (enhanced) | `eqdma_soft_access.c` | 2048 |
   | CPM4 | `qdma_cpm4_access.c` | version-specific |

**Queue lifecycle.** `add` (allocate qid, set mode and direction) → `start`
(build H2C/C2H descriptor rings, program the *queue context* into hardware,
program the interrupt context if in interrupt mode, attach a service thread) →
transfers → `stop` → `delete`. The "context" is the crucial concept: queue state
lives in hardware context RAM, programmed via an indirect
address/data/command register triple. Context programming failures are the most
common QDMA bring-up bug and they surface as queues that accept descriptors and
never complete.

**MM vs ST.**
- **MM (memory-mapped)**: the descriptor names a host address, a card AXI
  address, and a length. The driver writes descriptors and bumps the hardware
  producer index (PIDX). This is what a buffer `sync()` uses.
- **ST (streaming)**: H2C descriptors point at host buffers whose data is
  streamed out as AXI4-Stream; C2H uses a prefetch engine plus a separate
  **completion (CMPT) ring** that reports how much arrived and with what
  metadata. C2H ST is substantially more complex — prefetch contexts, buffer
  size rings, `qdma_st_c2h.c` — and is where most QDMA driver bugs live.

**Interrupt modes**: poll, direct (one MSI-X vector per queue), and aggregation
(an interrupt-ring coalescing many queues into one vector). Aggregation is what
makes thousands of queues viable.

**Error handling**: each HAL defines an error table mapping hardware error bits
to handlers, covering descriptor errors (poison, timeout, parameter mismatch),
RAM single/double-bit errors, and separate H2C/C2H datapath error classes. A
descriptor error usually means the *driver* built a bad descriptor, not that the
hardware failed — check length alignment and address validity first.

### 8.3 CPM5 specifics (V80)

CPM5 contains two PCIe controllers and two QDMA/bridge subsystem instances,
bridged to the rest of the device by high-bandwidth AXI interfaces into the NoC.
Known address facts for the AVED configuration:

- PCIe BAR0 → PL memory space `0x201_0000_0000`–`0x201_0FFF_FFFF` (256 MB).
- QDMA register base `0x6_1000_0000`; Bridge register base `0x6_0000_0000`.
- The **slave bridge** is the reverse path: it lets an AXI master *inside* the
  device issue reads/writes that become PCIe transactions to host memory. PG347
  documents limitations on its registers.

> **[field] The slave bridge is not universally functional.** On the V80
> *compute* shell used by SLASH, routing an HLS kernel's `m_axi` port to the
> `HOST` target (which resolves to `/qdma_slave_bridge_noc/S00_AXI`) produced an
> AXI master whose reads never returned. A byte-identical kernel with the same
> port routed to `HBM0` completed in under 0.1 s with correct results. The
> control experiment:
>
> | build | `sp=` target | AP_CTRL trace | result |
> |---|---|---|---|
> | `hostprobe` | `HOST` | stuck at `0x1` (ap_start) for ~15 min | never completes |
> | `hostprobe_hbm` | `HBM0` | `0x4` → `0x1` → `0xe` | `sum=524800`, correct |
>
> Neither the C-model emulator nor the RTL simulator can catch this: emulation
> shares process memory, and simulation copies host memory into the model. **A
> host-mastering path can only be validated on hardware.**

---

## 9. Runtime/driver stacks compared

### 9.1 XRT (Alveo U-series)

Two drivers: `xclmgmt` binds the management PF, `xocl` binds the user PF.

- `xclmgmt`: board recovery when compute units hang the AXI bus, sensor
  collection, **AXI Firewall** monitoring, clock scaling, power measurement,
  loading firmware onto embedded soft processors (ERT, CMC).
- `xocl`: XDMA/QDMA engine programming behind a buffer-migration API,
  multi-process context management, compute-unit execution pipeline via the
  hardware scheduler (ERT), interrupt handling, and address-remapper setup so
  kernels can reach host memory directly.

Platform model: a **static shell** loaded from PROM at cold boot (immutable
until the next cold boot) plus a **dynamic region** reprogrammed per xclbin. The
AXI Firewall is the notable defensive design — it sits between the shell and the
dynamic region and traps illegal AXI transactions before they can hang the whole
bus, so a buggy kernel costs a reprogram instead of a reboot.

### 9.2 AVED / AMI (V80 reference design)

AVED is AMD's Versal example design for the V80. Its host-side piece is **AMI**,
shipped in three parts: the `ami` kernel module, a userspace API library, and
`ami_tool`.

The interesting architectural piece is the **GCQ (Generic Command Queue)**: the
host and the card's RPU firmware (AMC) communicate through a region of shared
memory plus a mailbox IP. The GCQ implements *dual circular buffers* — one
submission, one completion — giving bidirectional command flow with status
reporting, plus interrupts in both directions so neither side has to poll. PDI
data destined for flash is transferred from the host into a DDR region mapped
into PF0's BAR, then written to OSPI by the AMC.

`ami_tool` surface worth knowing for debug:

```
ami_tool overview                     # device state, design name, AMI/AMC versions
ami_tool pcieinfo -d <BDF>            # link speed/width, NUMA node
ami_tool sensors  -d <BDF> -f json    # thermal/power
ami_tool bar_rd / bar_wr              # raw register access
ami_tool debug_verbosity -d <BDF> -l debug
echo 1 > /sys/bus/pci/drivers/ami/ami_debug_enabled
```

`AMC Heartbeat expired event received` in dmesg means the RPU firmware stopped
answering — the card needs a reset, and `ami_tool` commands that route through
GCQ will hang or fail.

### 9.3 SLASH (V80 compute platform)

Covered in depth in KB-3. Three PFs, a GPLv2 kernel module (`slash.ko`) driving
PF1 and PF2, an MIT userspace stack, and — unusually — a **daemon (`vrtd`)
between the application and the driver** that arbitrates multi-tenant access and
holds the privileged operations. That is a meaningful design departure from XRT,
which goes straight from library to ioctl.

### 9.4 The Linux FPGA subsystem (for completeness)

Upstream Linux has its own FPGA framework, which none of the three AMD stacks
above use, but which you will meet on embedded parts and Intel cards:

- **fpga-mgr** (`fpga-mgr.c`) — low-level drivers that know how to program a
  specific device.
- **fpga-bridge** — gates buses during programming to stop spurious signals
  escaping a region being reconfigured. Either hard bridges or soft "freeze"
  bridges in fabric.
- **fpga-region** (`fpga-region.c`) — associates managers and bridges into a
  reconfigurable region.
- **DFL (Device Feature List)** — Intel's discovery scheme: a walkable linked
  list of feature headers in BAR space, with an FME (FPGA Management Engine)
  driver that instantiates manager/bridge/region objects during PR feature init.

DFL's discovery-by-walking-headers is the same idea as AVED's VSEC metadata, and
is worth imitating regardless of stack.

---

## 10. Reconfiguration

Reprogramming an FPGA under a live OS is the operation that distinguishes FPGA
drivers from ordinary PCIe drivers. There are three mechanisms, in increasing
order of blast radius.

### 10.1 Partial reconfiguration

Only a fenced region of fabric changes; PCIe and the shell stay up. Requires
bridges/freeze logic around the region. Cheapest and safest, but requires the
design to have been floorplanned for it.

### 10.2 Function-Level Reset (FLR)

Resets a single PCIe function without touching the link. Sufficient to clear a
wedged DMA engine; insufficient to reload a bitstream.

### 10.3 Secondary Bus Reset (SBR / "hot reset")

Software sets, then clears, the `BUS_RESET` bit in the *bridge control register*
of the upstream bridge's config space. Every device below that bridge is reset
and loses all configuration. In Linux:

```c
pci_bridge_secondary_bus_reset(bridge);
```

which saves and restores bridge config state around the reset. Also exposed to
userspace as the bridge's `reset_bus` sysfs attribute.

On the V80 this is what triggers a PDI reload: AMI's hot reset writes `0x40` to
the bridge control register, which the card interprets as a reset and reloads
its image.

The full sequence — SLASH's `/dev/slash_hotplug` exposes exactly these
primitives:

```text
1. REMOVE      PF0, PF1, PF2      ← tear down all three functions
2. TOGGLE_SBR  on the root port   ← reset the FPGA, reload the bitstream
3. RESCAN                         ← re-enumerate the bus
4. HOTPLUG     each function      ← bind drivers to the new device
```

Two locking details from `slash_hotplug.c` worth stealing:

- `RESCAN`, `REMOVE`, and `HOTPLUG` all take `pci_lock_rescan_remove()`, which
  serialises them against each other and against the kernel's own hotplug.
- `TOGGLE_SBR` **drops that lock before calling
  `pci_bridge_secondary_bus_reset()`** to avoid deadlocking against the PCI slot
  lock. It holds the lock only across `pci_find_bus()` + `pci_dev_get()`.
- It then sleeps **1000 ms** after deassertion. The PCIe spec minimum is 100 ms;
  real FPGA endpoints need much more, and the extra margin also covers the
  kernel-internal window between reset and the link actually being usable.

---

## 11. Error handling: AER

PCIe Advanced Error Reporting classifies uncorrectable errors as **non-fatal**
(the transaction is unreliable, the link is fine) or **fatal** (the link is
unreliable). Linux's AER driver responds to uncorrectable errors by performing a
Secondary Bus Reset at the port above the originating device.

Registers of interest per device: the uncorrectable error *status*, *mask*, and
*severity* registers. The severity register decides which uncorrectable errors
are reported as fatal — and AER-aware drivers are permitted to reprogram it.

> **[field] This is the knob that matters when a link event takes down the
> host.** On this machine, a `TOGGLE_SBR` on root port `0000:00:01.1` hard-reset
> the entire host with no MCE record written — the last line in the journal is
> `slash_hotplug: ioctl: TOGGLE_SBR succeeded`, followed by a new boot. The same
> SBR succeeded three times in an earlier boot, so it is intermittent, which
> makes it a race rather than a deterministic fault.
>
> The underlying policy is firmware's: a fatal PCIe error on this root port is
> escalated to a platform reset instead of being logged. The mitigations, in
> order of preference:
>
> 1. **BIOS**: set PCIe uncorrectable/fatal error severity to non-fatal, or
>    disable "System Error on PCIe fatal". On AMI/AMD boards this lives under
>    *AMD CBS → NBIO*, or as *PCIe AER Support* / *System Error Severity*.
> 2. **Unbind every driver on the bus before any link transition.** AMD's own
>    code says so, in `vrt/vrtd/src/reset.c:334`: *"If any function remains
>    bound while the bus is reset, the kernel may attempt MMIO or config-space
>    accesses to a device whose link is down, which can cause machine checks or
>    system hangs."*
> 3. Avoid the reset entirely — see §12.3.

---

## 12. Field hazards **[field]**

Findings from this bring-up that are not in any vendor document.

### 12.1 A design write succeeds only on a freshly reset device

Measured on 2026-08-18:

```
20:29:58  design_write: shell switch required current=0 required=2
20:29:59  reset_with_ami: AMI_IOC_DEVICE_BOOT(partition=1) OK
20:29:59  removed 01:00.0/.1/.2  →  SBR  →  rescan
20:30:18  Design write completed successfully          ← load #1 OK
20:33:53  Design write submitted (3m35s later)
20:34:03  Failed to transfer design writer payload: Input/output error
```

The daemon runs `reset_with_ami` **only when the requested shell differs from
the current one**. Once the card reports `Shell: compute`, that condition is
never true again, so no reset happens and every subsequent load fails. A failed
load also drives the AMC to `NO_AMC`, and recovering *that* requires JTAG —
because `AMI_IOC_DEVICE_BOOT` reaches the AMC over GCQ, and a wedged AMC cannot
service its own recovery.

**The fix that makes iteration affordable:** skip reprogramming when the design
is already resident.

```cpp
vrt::Device device(bdf, vbin, /*program=*/false);
```

Many runs then share one reset instead of each run consuming one.

### 12.2 Runtime hazards in the vendor library

- **`vrt::Kernel::wait()` blocks forever** — there is no timeout parameter. Poll
  the HLS `AP_CTRL` register (offset `0x00`: bit0 `ap_start`, bit1 `ap_done`,
  bit2 `ap_idle`, bit3 `ap_ready`) with your own bound instead, or a stuck
  kernel yields nothing but a `Killed` from your outer timeout.
- **`std::cout` is fully buffered under `runuser`/`tee`.** Progress prints must
  go to `std::cerr` or they vanish when the process is killed. This turned a
  15-minute diagnostic run into zero output, twice.
- **The device allocator has a minimum block size.** VRT's
  `MediumBlockSuperblock : BuddySuperblockBase<12, 21>` means the smallest
  allocation it will serve is 2^12 = 4096 bytes. Requesting 256 words throws
  `Size too small for MediumBlockSuperblock`.

### 12.3 The recovery loop trap

The four steps form a cycle that is easy to walk into:

1. Plain warm reboot → card loses the POST race → root port `00:01.1` absent.
2. JTAG-load the shell to fix enumeration → firmware now reports `Shell:
   unknown`.
3. First PDI load therefore requires a **shell switch** → `reset_with_ami` →
   `TOGGLE_SBR` → intermittent host hard reset.
4. → back to step 1.

The way out is to break step 3: get the card to a known shell *once*, then never
program again in that session (`program=false`), and never JTAG-load unless the
card is actually off the bus.

### 12.4 Diagnostic mistakes worth naming

- **`pgrep -f <pattern>` matches its own command line.** A "is the build still
  running?" check will report yes forever. Use `pgrep -f -- "$pat" | grep -v $$`
  or check for the actual artefact.
- **`ps` `ELAPSED` is `MM:SS` for short-lived processes**, not minutes. A
  recovery that was 40 s in was misread as 28 minutes.
- **Appended log files defeat completion greps.** If a script `>>`s to a log,
  `grep DONE` matches yesterday's run. Emit a run-unique marker.
- **A truncated `.so` looks up-to-date to `make`.** A build interrupted by a
  host reset left a 0-byte `libvortex-aved.so` with a fresh mtime; `make` then
  had nothing to do. `rm` the artefact when a build is interrupted abnormally.

---

## 13. Debug toolbox

| Question | Tool |
|---|---|
| Is the card on the bus? | `lspci -d 10ee: -nn`, `lspci -tv` |
| Does the root port exist at all? | `ls /sys/bus/pci/devices/0000:00:01.1` |
| Link speed/width negotiated? | `lspci -vv -s <BDF>` → `LnkSta:`; `ami_tool pcieinfo` |
| Which driver is bound? | `ls -l /sys/bus/pci/devices/<BDF>/driver` |
| BAR sizes and addresses? | `lspci -vv`, or `GET_BAR_INFO` ioctl |
| AER errors accumulating? | `lspci -vv` → `UESta:`/`CESta:`; `dmesg \| grep -i aer` |
| Did the host crash or shut down? | `journalctl --list-boots`; a clean boot ends with `systemd-shutdown: Shutting down`, a crash just stops |
| Machine check? | `journalctl -b -1 \| grep -i mce`; `ras-mc-ctl --errors` |
| Card firmware alive? | `ami_tool overview` → state `READY` vs `NO_AMC`; dmesg heartbeat messages |
| Fabric-level debug without a cable | XVC over PCIe (`/dev/xdma<N>_xvc`) + Vivado Hardware Manager |
| Kernel-side coverage of your driver | build with `GCOV=1` on a `CONFIG_GCOV_KERNEL=y` kernel, `lcov` + `genhtml` |
| DMA correctness/bandwidth | `v80-smi validate` (pattern `i ^ seed` integrity + H2C/C2H bandwidth) |

---

## 14. Checklist for a new driver

**Correctness**
- [ ] `MODULE_DEVICE_TABLE` present, so autoload works.
- [ ] `probe()` rejects unexpected PFs.
- [ ] `pci_set_master()` called before any device-initiated DMA.
- [ ] DMA mask set before the first mapping call.
- [ ] Userspace node created last in probe, destroyed first in remove.
- [ ] Every polling loop treats `readl() == ~0` as "device gone", not as data.
- [ ] `wmb()` between writing a descriptor and ringing its doorbell.
- [ ] `remove()` is safe against an already-absent device.

**ABI**
- [ ] Every ioctl struct leads with `__u32 size`; kernel copies `min()` both ways.
- [ ] Fixed-width types, explicit padding, reserved fields validated as zero.
- [ ] Unknown ioctl → `-ENOTTY`.
- [ ] A `GET_DEVICE_INFO`-equivalent exists and the docs say node numbers are
      not stable across rescan.
- [ ] fd-returning ioctls document that the fd is the *return value*.

**Reconfiguration**
- [ ] Nothing is cached across a reprogram.
- [ ] All drivers unbind before any link transition.
- [ ] SBR path drops `pci_lock_rescan_remove()` before the reset call.
- [ ] Post-SBR settle is ≥ 1 s, not the spec minimum of 100 ms.
- [ ] There is a documented recovery path for "the card is off the bus", and it
      states whether it needs JTAG or a cold power cycle.

**Operability**
- [ ] Module parameters for thread counts and debug paths.
- [ ] A verbosity control that does not require a rebuild.
- [ ] Log lines that name the operation and the BDF, so a journal can be read
      after the fact by someone who was not there.

---

## Sources

- [AVED Overview — AVED documentation](https://xilinx.github.io/AVED/latest/AVED+Overview.html)
- [AVED — Host to Card Communication](https://xilinx.github.io/AVED/amd_v80_gen5x8_exdes_2_20240408/AVED+-+Host+to+Card+Communication.html)
- [AVED — Device Programming](https://xilinx.github.io/AVED/amd_v80_gen5x8_exdes_2_20240408/AVED+-+Device+Programming.html)
- [AVED V80 — CIPS Configuration](https://xilinx.github.io/AVED/amd_v80_gen5x8_exdes_2_20240408/AVED+V80+-+CIPS+Configuration.html)
- [AVED Debug Techniques](https://xilinx.github.io/AVED/amd_v80_gen5x8_24.1_20241002/AVED+Debug+Techniques.html)
- [AMI — Hot Reset](https://xilinx.github.io/AVED/amd_v80_gen5x8_24.1_20241002/AMI+-+Hot+Reset.html)
- [AMI Architecture](https://xilinx.github.io/AVED/amd_v80_gen5x8_exdes_1_20231204/AMI+Architecture.html)
- [Xilinx/dma_ip_drivers on GitHub](https://github.com/Xilinx/dma_ip_drivers)
- [QDMA Linux Driver Architecture (DeepWiki)](https://deepwiki.com/Xilinx/dma_ip_drivers/2.1.1-qdma-linux-driver-architecture)
- [XDMA Linux Kernel Driver (DeepWiki)](https://deepwiki.com/Xilinx/dma_ip_drivers/3.1-xdma-linux-kernel-driver)
- [Xilinx QDMA Linux Driver documentation](https://xilinx.github.io/dma_ip_drivers/master/QDMA/linux-kernel/html/index.html)
- [How To Write Linux PCI Drivers — kernel.org](https://docs.kernel.org/PCI/pci.html)
- [Bus-Independent Device Accesses — kernel.org](https://www.kernel.org/doc/html/latest/driver-api/device-io.html)
- [The PCI Express Advanced Error Reporting Driver Guide HOWTO](https://docs.kernel.org/PCI/pcieaer-howto.html)
- [FPGA Device Feature List (DFL) Framework Overview](https://docs.kernel.org/fpga/dfl.html)
- [FPGA Region — kernel.org](https://www.kernel.org/doc/html/v5.5/driver-api/fpga/fpga-region.html)
- [XRT and Vitis Platform Overview](https://xilinx.github.io/XRT/master/html/platforms.html)
- [Alveo Platform Loading Overview — XRT](https://xilinx.github.io/XRT/master/html/platforms_partitions.html)
- [Slave Bridge Registers Limitations — PG347](https://docs.amd.com/r/en-US/pg347-cpm-dma-bridge/Slave-Bridge-Registers-Limitations)
- [PCIe_CPM Lab 2: QDMA AXI MM Interface to NoC and DDR](https://github.com/Xilinx/PCIe_CPM/blob/main/docs/Lab2/Lab2.md)
- [Controlling Hardware — UG1399 (Vitis HLS)](https://docs.amd.com/r/2020.2-English/ug1399-vitis-hls/Controlling-Hardware)
- [PCIe Hot Reset on Linux — Alex Forencich](https://alexforencich.com/wiki/en/pcie/hot-reset-linux)
