# Publish Plan: SLASH fork → toolchain prebuilt → vortex gfxw_v2

Date: 2026-08-28
Status: EXECUTED (pushes pending credentials)

Decisions taken at execution time (user-directed):
- The SLASH publish branch is **`vortex_3.x`** (renamed from `vortex/v80-support`).
- The prebuilt lands on the **`release-v3.0.1`** branch and the existing
  **`v3.0.1` tag is refreshed** to it (the repo's established convention —
  its history is a chain of "refresh prebuilt for the v3.0.1 tag" commits)
  instead of cutting a new `v3.1`; vortex's `VERSION` pin moves
  `v3.0 → v3.0.1`.
- The qdma `pr_fmt` fix needed no new commit: it was already captured as an
  in-tree build-time patch (`driver/patches/0003-libqdma-pr-fmt-guard.patch`,
  applied by the driver Makefile's `libqdma-patches` target); the submodule
  edit was just applied build state and has been reverted to pristine.

## 1. Goal

Publish everything produced during the V80 bring-up, in an order where each
push only references things that already exist publicly:

1. **SLASH** (the V80 platform fork) — the source of the userspace we ship.
2. **vortex-toolchain-prebuilt** — gains a `slash` tarball built *from that
   pushed fork*, under a new toolchain revision tag.
3. **vortex `gfxw_v2`** — the 89 bring-up commits, whose
   `toolchain_install.sh --slash` downloads *that pushed tarball*.

Reversing any pair breaks a consumer: a `gfxw_v2` clone would try to fetch a
tarball that isn't published yet, and a published tarball would have no public
source it was built from.

## 2. Current state (verified 2026-08-28)

### 2a. SLASH — `~/dev/SLASH` (+ worktree `~/dev/SLASH-compute`)

Remote `origin` is upstream `Xilinx/SLASH`; we have **no fork remote yet**.
Two branches carry our work:

| Branch | Base | Local commits | Role |
|---|---|---|---|
| `vortex/v80-support` (worktree `~/dev/SLASH-compute`) | `origin/feature/compute_only_platform_v2` | 8 | What the machine's `.deb` packages and the prebuilt tarball are built from |
| `feature/host-buffer-allocator` | `origin/main` | 3 | Upstream-PR candidate: same fixes rebased onto `main` |

`vortex/v80-support` commits: Secure-Boot key enrolment, DKMS libqdma path
fix, debhelper/vrtd.postinst, dh-dkms build-dep, slashkit Vivado-version and
sim-memory-model fixes, DMA-coherent host buffers, kernel-compat shims.
`feature/host-buffer-allocator` commits: QDMA-on-6.15 fix, host-buffer
allocator, Vivado 2025.2 linker acceptance.

**Loose ends that must be resolved before pushing:**

1. **Uncommitted fix inside the `qdma_drv` submodule** (`submodules/qdma_drv`
   → `Xilinx/dma_ip_drivers`): an 11-line guard in
   `QDMA/linux-kernel/driver/libqdma/qdma_platform_env.h` that re-arms the
   kernel default `pr_fmt` (needed because our compat header `#undef`s it).
   Pushing SLASH does **not** carry submodule content. The installed `.deb`
   works only because dpkg-buildpackage vendored the local checkout.
2. **Untracked `slash_deps.sh`** in `~/dev/SLASH` — the dependency installer
   we wrote; belongs in the fork so a fresh clone can build.
3. **Dirty `AVED` submodule** in `~/dev/SLASH` — only a build-generated
   `ami_driver_version.h` and object files; restore, don't commit.
4. Build artifacts (`.o.cmd` files) inside `qdma_drv` — clean, don't commit.

### 2b. vortex-toolchain-prebuilt — not cloned locally

`github.com/vortexgpgpu/vortex-toolchain-prebuilt` exists (branch `master`,
tags `v2.3`, `v3.0`, `v3.0.1`). Vortex's `VERSION` file pins
`TOOLCHAIN_REV=v3.0`, and `toolchain_install.sh` fetches
`raw/${TOOLCHAIN_REV}/slash/${OSVERSION}/slash.tar.bz2`. No `slash` component
exists there yet, and tag `v3.0` is immutable — publishing requires a **new
commit on `master` plus a new tag**.

### 2c. vortex — `~/dev/vortex_gfxw_v2`, branch `gfxw_v2`

**89 commits ahead** of `origin/gfxw_v2`, working tree clean, remote is
`https://github.com/vortexgpgpu/vortex.git`. The branch already contains
`toolchain_install.sh.in --slash` (downloads the tarball, documents
`VRT_HOME`), but `ci/toolchain_prebuilt.sh.in` (the *packaging* script) has
no `slash()` function yet — the tarball would be hand-rolled and
irreproducible. Two small commits are still needed (see Stage 3).

### 2d. Blocking prerequisite: credentials

This machine has no push credentials at all: HTTPS remotes with no helper or
token, no `gh`, no local private keys, and the VSCode-forwarded SSH agent
reports *no identities*. **Nothing below can start until one of:**

- you run `ssh-add` on your **local** machine (the forwarding then carries
  your key here — the fastest option), or
- you provide a GitHub personal-access token for one-time use, or
- you run the pushes yourself from an authenticated machine.

Additionally, `vortexgpgpu/SLASH` must exist as a fork of `Xilinx/SLASH`
(it isn't publicly visible today). Creating it is a one-click GitHub action
that needs your org permissions; I cannot do it from here.

## 3. Stage 1 — SLASH

1. **Fold the qdma_platform_env.h fix into SLASH itself** so the submodule
   stays pristine at the upstream `dma_ip_drivers` commit. The fix guards
   against our own compat header's `#undef pr_fmt`; the natural home is that
   same compat header (re-define the kernel default after the `#undef`),
   with the submodule change reverted. Falls back to SLASH's existing
   source-patch mechanism if header ordering makes that impossible. One
   commit on `vortex/v80-support`; verify by rebuilding the DKMS driver
   against the running kernel and confirming zero diff inside the submodule.
2. **Commit `slash_deps.sh`** onto `vortex/v80-support`.
3. **Clean the checkouts**: restore the AVED submodule's generated file,
   delete stray build artifacts (nothing committed from either submodule).
4. **Add the fork remote and push both branches:**
   ```
   git remote add fork git@github.com:vortexgpgpu/SLASH.git
   git push fork vortex/v80-support feature/host-buffer-allocator
   ```
5. *(Optional, later)* open the upstream PR: `feature/host-buffer-allocator`
   → `Xilinx/SLASH main`.

**Exit criteria:** both branches visible on the fork; a scratch clone of the
fork at `vortex/v80-support` builds the userspace with no local edits.

## 4. Stage 2 — vortex-toolchain-prebuilt

1. Clone `vortexgpgpu/vortex-toolchain-prebuilt` (`master`).
2. **Build the tarball from the pushed fork, not from the working tree**: a
   scratch clone of `vortexgpgpu/SLASH@vortex/v80-support`, userspace-only
   build into a clean prefix (the `~/dev/.slash-local` recipe), then package
   the prefix as a top-level `slash/` directory (`bin lib include share`) —
   exactly what the installer expects to `mv` into `$TOOLDIR/slash` and use
   as `VRT_HOME`.
3. Place it at **`slash/ubuntu/focal/slash.tar.bz2`**. Vortex's `configure`
   deliberately maps focal/jammy/noble all to `ubuntu/focal`, so this is the
   path a 24.04 machine requests. Caveat recorded in the README: this binary
   is *built on* noble and needs noble-era glibc; focal/jammy users must
   build SLASH from source. If the compressed tarball exceeds 100 MB
   (GitHub's raw-file limit) it gets split into 50 MB parts like the other
   components; expected size (~100 MB installed prefix) should compress well
   under that.
4. Commit on `master`, **tag `v3.1`**, push both. The tag's tree still
   contains every existing component, so bumping vortex's pin is safe for
   all of them.

**Exit criteria:**
`wget https://github.com/vortexgpgpu/vortex-toolchain-prebuilt/raw/v3.1/slash/ubuntu/focal/slash.tar.bz2`
succeeds and the extracted tree matches the built prefix.

## 5. Stage 3 — vortex `gfxw_v2`

1. New commit: **`slash()` packaging function in
   `ci/toolchain_prebuilt.sh.in`** (mirror of the other components: tar the
   prefix, split if needed) so Stage 2 is reproducible next time.
2. New commit: **bump `VERSION` to `TOOLCHAIN_REV=v3.1`** — only after the
   `v3.1` tag from Stage 2 is live, since the pin changes the fetch URL for
   every component.
3. Commit this document.
4. Push:
   ```
   git push origin gfxw_v2
   ```
   (~92 commits: RTL CP/AFU fixes, the staged-sync ownership model, reset
   architecture + acceptance, aved auto-discovery/no-program defaults,
   tests, docs.)

**Exit criteria:** on a scratch checkout of pushed `gfxw_v2`:
`./configure && ci/toolchain_install.sh --slash` into a scratch `TOOLDIR`
yields a working `VRT_HOME`, and the aved runtime builds against it.

## 6. What this plan does NOT include

- No synthesis/Vivado builds (rel1t/rel2 stay stopped unless you ask).
- No pushing of bitstreams/vbins — `rel1` stays a local artifact for now.
- No upstream `Xilinx/SLASH` PR yet (listed as optional in Stage 1).
- No changes to the installed `.deb` stack on this machine — it already
  matches `vortex/v80-support`.

## 7. Risks / open decisions

| Risk | Mitigation |
|---|---|
| `vortexgpgpu/SLASH` fork may need org-admin rights to create | You create it once; everything else is push-only |
| qdma fix can't move out of the submodule cleanly | Fallback: build-time patch shipped in SLASH (mechanism already exists for the RHEL patch) |
| Tarball under `ubuntu/focal` but built on noble | Documented in prebuilt README; source build remains the focal/jammy path |
| `TOOLCHAIN_REV` bump affects all components | New tag is a superset of `v3.0`; verified by the Stage 3 exit criterion |
| Push rejected (branch protection / non-fast-forward on `origin/gfxw_v2`) | `git fetch` first; if upstream moved, rebase is your call before pushing |

## 8. Execution checklist

| # | Action | Who |
|---|---|---|
| 0 | `ssh-add` on your local machine (or provide a token) | **you** |
| 0b | Create fork `vortexgpgpu/SLASH` | **you** |
| 1 | De-submodule the qdma fix, commit `slash_deps.sh`, clean trees | me |
| 2 | Push both SLASH branches to the fork; scratch-clone build check | me |
| 3 | Build + package `slash.tar.bz2` from the fork; commit, tag `v3.1`, push prebuilt | me |
| 4 | `slash()` packager + `VERSION` bump + this doc committed on `gfxw_v2` | me |
| 5 | `git push origin gfxw_v2`; scratch-checkout `--slash` install check | me |
