# C++ Coding Guidelines

Keep your code warning-free, consistent and easy to read.

## 1. Indentation
- Use **2 spaces** per indent level.
- **Do not** use tab characters.

```cpp
if (condition) {
  doSomething();
}
```

## 2. Brace Placement
- Opening braces go on the **same line** (K&R style).
- Closing braces go on their **own line**, aligned with the start of the declaration.

```cpp
class MyClass {
public:
  void foo() {
    // ...
  }
};
```

- **Braces are mandatory** on **every** `if`, `else if`, `else`, `for`,
  `while`, and `do` body — **even when the body is a single statement**.
  The brace-less shortcut is forbidden because:
    - Adding a second statement to the branch silently re-scopes the
      first to be unconditional (the appended statement falls outside
      the implicit one-line body). Same hazard as Verilog single-line
      `if`-without-`begin/end`. Goto-fail-class bug.
    - Diff hygiene: changing a one-liner into a multi-statement block
      produces a noisy multi-line diff that obscures the actual change.

```cpp
// BANNED — brace-less shortcut
if (cond) doSomething();
else      doOther();

for (int i = 0; i < N; ++i) work(i);

// REQUIRED — always braces
if (cond) {
  doSomething();
} else {
  doOther();
}

for (int i = 0; i < N; ++i) {
  work(i);
}
```

## 3. Spaces
- **One space** after keywords (`if`, `for`, `while`, `switch`).
- **No space** before function call parentheses.
- **Spaces** around binary operators.

```cpp
int x = a + b;
if (x == 0) {
  x = 1;
}
foo(x);
```

## 4. Function Declarations and Definitions
- In headers, parameter names may be omitted if unused.
- In definitions, **align** parameters vertically when **multi-line**.

```cpp
void foo(int a, int b, int c, int d, int e, int f, int g, int h
         int i);
```

## 5. Constructor Initializer Lists
- Place each initializer on its **own line** if the list is multi-line.
- Align subsequent lines under the colon.

```cpp
MyClass::MyClass(int a, int b)
    : a_(a)
    , b_(b) {
}
```

## 6. Comments
- Use `//` for **single-line** comments.
- Reserve `/* ... */` for **block** comments sparingly.
- Use Doxygen-style for **public API**:

```cpp
/// Computes the foo of bar.
/// @param x The input value.
/// @return The computed result.
int foo(int x);
```

## 7. Using #ifdef
- Preserve indent of nested code and shift pre-processor left by one level

Base function (before):
```cpp
regno_t to_regno_base(const reg_t& reg, bool has_type, bool is_dp) {
  if (has_type) {
    if (is_dp) {
      return regno_t{reg.rtype, reg.id, reg.group};
    }
    return regno_t{reg.rtype, reg.id};
  }
  return regno_t{reg.id, 0};
}
```

Adding ifdef (after):
```cpp
regno_t to_regno_base(const reg_t& reg, bool has_type, bool is_dp) {
  if (has_type) {
  #ifdef
    if (is_dp) {
      return regno_t{reg.rtype, reg.id, reg.group};
    }
  #else
    __unused(is_dp);
  #endif
    return regno_t{reg.rtype, reg.id};
  }
  return regno_t{reg.id, 0};
}
```

## 8. Source-Tree Layering — `sw/` ↔ `hw/`/`sim/` Bidirectional Isolation

Vortex's source tree separates the **software stack** (`sw/`) from the
**hardware** (`hw/`) and **simulator** (`sim/`) implementations. The
isolation is **bidirectional**:

- Files under `sw/kernel/` and `sw/runtime/` **MUST NOT** `#include`
  or otherwise reference anything in `hw/*` or `sim/*`.
- Files under `sim/*` and `hw/*` **MUST NOT** reference anything in
  `sw/kernel/` or `sw/runtime/`.

```cpp
// FORBIDDEN — sw/runtime reaching into sim/
#include "../sim/simx/processor.h"

// FORBIDDEN — sw/kernel reaching into hw/
#include "../hw/rtl/VX_config.vh"

// FORBIDDEN — sim/ reaching into sw/kernel/
#include "vx_graphics.h"   // (from sw/kernel/include/)

// FORBIDDEN — sim/ reaching into sw/runtime/
#include "graphics.h"      // (from sw/runtime/include/)
```

Build-system equivalents are equally forbidden — neither
`sim/*/Makefile` nor `hw/*/Makefile` may add
`-I.../sw/kernel/include` or `-I.../sw/runtime/include` to its
compile flags.

### Communication channel between layers — `sw/common/`

`sw/common/` is the **vortex-internal shared layer** accessible from
all four layers (`sw/kernel`, `sw/runtime`, `sim/*`, `hw/*` via
inclusion in `sim/`-side build flags). It is never installed and
never visible to downstream consumers. Use it for:

| Need                                                | Location |
|-----------------------------------------------------|---------------------------|
| On-wire ABI structs (host writes / hardware reads)  | `sw/common/`              |
| Host-side hardware mirror models                    | `sw/common/`              |
| Shared host-side helpers (mem_alloc, bitmanip, …)   | `sw/common/`              |
| Generated build configuration                       | `build/sw/VX_types.h` (also shared) |

### Installed headers stay self-contained

The two installed include directories
([`sw/kernel/include/`](../sw/kernel/include/) and
[`sw/runtime/include/`](../sw/runtime/include/)) must not transitively
pull anything from `sw/common/`, `hw/*`, or `sim/*` — only stdlib +
the generated [`VX_types.h`](../sw/kernel/include/) +
sibling vortex public headers.

### CI enforcement

The bidirectional `sw/` ↔ `sim/` isolation is enforced mechanically
by [`ci/check_sw_sim_boundary.sh`](../ci/check_sw_sim_boundary.sh),
invoked at the top of `ci/regression.sh`. The script scans `sim/`
sources for any reference to `sw/{kernel,runtime}/include/` paths
or headers, and vice versa. The audit is rule-based, not policy-based:
every `#include` in `sw/{kernel,runtime}/include/*.h` must resolve
under those two directories (post-install) or stdlib, and no
`-I` flag in `sim/*/Makefile` may point at `sw/{kernel,runtime}/`.

## 9. Debug/Trace Macros
- **arguments inside `DP`, `DPN`, `DPH`, `DT`, `DTN`, `DTH` must be comma-separated**.

Correct:
```cpp
DT(2, "req: wid=" << wid << ", pc=0x" << std::hex << pc << std::endl);
```

Incorrect (space-separated entries):
```cpp
DT(2, "req: wid=" << wid << " pc=0x" << std::hex << pc << std::endl);
```

## 10. Comment Content & Intent

Comments describe what the adjacent code does and why, not the process that produced it. Prefer self-documenting code — good abstractions and consistent naming — and drop comments on code whose intent is already obvious; keep the rest brief, one or two lines per block as the norm (longer only where genuinely warranted, at the author's discretion), since over-detailed comments obscure the code and drift out of sync with later changes. Never embed development metadata or history (phase/step/version/part/feature/bug numbers, "proposal", "spec"), debugging or change narration ("fixing bug…", "was broken because…" — that is what commit messages are for), or references to design documents. Comments and names must not reference the other implementation layer's internals: host-side models (SimX, runtime, drivers) must not name RTL signals or parameters, and RTL must not name host-side/SimX details. The layers evolve independently, so any such reference silently goes stale. These rules apply to every source file and script.