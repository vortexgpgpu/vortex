# DTCU_Control commit 누락 수정 리포트

- 작성일: 2026-06-16
- 수정 파일: `sim/simx/core.cpp`
- 관련 FU: `FUType::DTCU_Control` (DTENSOR_START / DTENSOR_POLL)

---

## 1. 뭐가 문제였나

`Core`의 두 군데에서 `FUType` 별로 `switch`를 돌리는데, 둘 다 마지막이 `default: assert(false)` 이고 **`DTCU_Control` case가 빠져 있었음.**

- `Core::execute()` 의 functional-unit stall 집계 switch (`core.cpp:425` 부근)
- `Core::commit()` 의 instruction-mix 집계 switch (`core.cpp:463` 부근)

즉 `DTENSOR_START` / `DTENSOR_POLL` 명령이 commit 되거나, `DtcuControlUnit` 이 stall 될 때 `default: assert(false)` 로 빠짐.

```
switch (trace->fu_type) {
case FUType::ALU: ...; break;
case FUType::FPU: ...; break;
case FUType::LSU: ...; break;
case FUType::SFU: ...; break;
case FUType::TCU: ...; break;   // EXT_TCU_ENABLE
case FUType::VPU: ...; break;   // EXT_V_ENABLE
default: assert(false);          // <-- DTCU_Control 이 여기로 빠짐
}
```

## 2. 왜 지금까지는 "잘 됐었나"

맞게 추측하셨음 — **DEBUG 빌드가 아니었기 때문**.

- 기본/실행 빌드 플래그: `-O2 -DNDEBUG` (`sim/simx/Makefile:50`).
- `NDEBUG` 가 정의되면 `assert(...)` 는 **no-op** 이라 `default: assert(false)` 도 그냥 통과함.
- 그래서 릴리스 빌드에서는 DTCU 명령이 정상 commit 되고 결과도 정상이었음.
- 반대로 `DEBUG=...` (`-g -O0`, assert 활성) 로 빌드하면, **첫 DTENSOR 명령 commit 시점에 `assert(false)` 로 abort** 함. (gdb 로 디버깅하려고 DEBUG 빌드하면 그때 터짐.)

추가로, 릴리스에서도 `DTCU_Control` 명령이 per-FU 카운터(`tcu_instrs` 등) 어디에도 안 잡혔음. (전체 `perf_stats_.instrs` 에는 포함됨.)

## 3. 어떻게 고쳤나

두 switch 각각 `EXT_TCU_ENABLE` 블록 안, `TCU` case 바로 뒤에 `DTCU_Control` case 를 추가함.

`Core::execute()` (stall 집계):
```cpp
#ifdef EXT_TCU_ENABLE
  case FUType::TCU: ++perf_stats_.tcu_stalls; break;
  case FUType::DTCU_Control: break; // control-only; no separate stall bucket
#endif
```

`Core::commit()` (instruction-mix 집계):
```cpp
#ifdef EXT_TCU_ENABLE
  case FUType::TCU: ++perf_stats_.tcu_instrs; break;
  case FUType::DTCU_Control: break; // counted in total instrs; no separate bucket
#endif
```

`break;` 만 둔 이유:
- `DTENSOR_POLL` 은 kernel busy-wait loop 안에서 매우 많이 commit 됨. 이걸 `tcu_instrs` 로 세면 in-core TCU 명령 수가 poll 횟수로 오염됨.
- 그래서 control 명령은 별도 per-FU bucket 에 넣지 않고, 전체 `instrs` 에만 반영되게 둠 (assert 회피가 목적).

## 4. 이제 뭐가 달라지나

- **DEBUG 빌드(`-O0`, assert 활성)에서 DTCU 를 돌려도 abort 안 함.** → gdb 디버깅 가능.
- **릴리스 빌드 동작/숫자 결과는 변화 없음** (원래 assert 가 no-op 이라 통과하던 경로라서).
- per-FU 카운터는 여전히 `DTCU_Control` 을 따로 집계하지 않음 (전체 instrs 에는 포함).

## 5. 후속(선택)

DTCU control 명령을 따로 보고 싶으면 전용 카운터(`dtcu_instrs` 등)를 추가할 수 있음. 단 이건 `VX_types.h` CSR, `PerfStats`, `mpm_query` 플러밍까지 손대야 해서 별도 작업으로 분리하는 게 맞음. (현재는 불필요.)
