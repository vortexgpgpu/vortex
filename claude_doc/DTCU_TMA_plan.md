# DTCU TMA 구현 계획 (검토/승인용)

- 작성일: 2026-06-16
- 대상: SimX `DTensorCore` (RTL 아님)
- 방향: **Option C — DTCU 내부 자율 K-tile prefetch engine** (kernel API는 `dtensor_start`/`dtensor_poll` 그대로 유지)
- 목표: K-tile operand load와 compute를 overlap 시켜 `dtcu_wait_for_tma_cycles`(compute가 다음 tile 기다리는 시간)를 줄인다.

---

## 0. 현재 코드 구조 요약 (기준점)

### FSM (`d_tensor_core.h:74-84`, `tick()` = `d_tensor_core.cpp:945-1070`)
```
IDLE → DESC_REQ → DESC_WAIT → OP_REQ → OP_WAIT → EXECUTE → (K 남으면 OP_REQ) → OUT_REQ → OUT_WAIT → (tile 남으면 OP_REQ) → DONE
```
- 현재는 **완전 serial**: K-tile load(OP_REQ/OP_WAIT) → EXECUTE → 다음 K load → EXECUTE …

### 버퍼 (`d_tensor_core.h:109-111`, 할당 `d_tensor_core.cpp:223-225`)
- `a_buf_` (tile_m_*8), `b_buf_` (8*tile_n_), `accum_buf_` (tile_m_*tile_n_) — **단일 버퍼**.

### 메모리 타이밍 모델 (`d_tensor_core.cpp:863-942`)
- `build_req_lists_()`가 현재 tile_k_idx_에 대한 operand cache line 리스트(`op_req_lines_`)와 (마지막 K일 때) output 리스트(`out_req_lines_`)를 만든다.
- `issue_next_op_req_()`/`issue_next_out_req_()`가 **한 번에 한 cache line**씩 `mem_req_out`으로 발행하고 `pending_tag_` 하나로 응답을 기다린다 (= outstanding 1개).
- functional 데이터는 `load_operands()`/`store_output()`에서 timing 요청 완료 후 `ram_->read/write`로 직접 처리.

### compute 타이밍 (`d_tensor_core.cpp:284-310, 1002-1024`)
- `estimate_execute_cycles_()` = equivalent WMMA micro-op 수 × 4 cycle. `EXECUTE`에서 `exec_cycles_left_`를 매 tick 감소 → 0이면 `execute_mma()`.
- ⇒ compute가 멀티사이클이라 **overlap이 cycle로 측정 가능**.

---

## 1. 먼저 승인이 필요한 설계 결정

| # | 결정사항 | 제안(기본값) | 비고 |
|---|---|---|---|
| D1 | TMA staging 위치 | **DTCU-private ping-pong buffer** | 현재 `a_buf_/b_buf_` 확장. shared memory(Option B)는 후속 |
| D2 | kernel API | **불변** (`dtensor_start`/`poll`) | 자율 prefetch라 변경 불필요 |
| D3 | L2 포트 | **기존 DTCU 포트 공유** | 별도 포트는 counter/동시성엔 깔끔하나 invasive. 우선 공유 |
| D4 | 메모리 outstanding | **outstanding 1개 유지 + compute 중 prefetch 진행** | EXECUTE 동안엔 main FSM이 mem 안 쓰므로 prefetch가 포트 점유 가능. multi-outstanding은 후속 |
| D5 | output store overlap | **Phase 5로 보류** | load overlap 먼저 (1순위) |
| D6 | compute 분할 | **현행 유지** (K-tile 단위 1회 execute_mma + cycle budget) | budget이 prefetch와 겹칠 시간 표현 |

→ 위 기본값으로 진행해도 될지, 바꿀 항목이 있는지 알려주세요.

---

## 2. Phase별 상세 계획

### Phase 0 — Baseline & robustness (거의 완료)
- [x] `DTCU_Control` commit/stall switch 누락 수정 (`core.cpp`) — DEBUG 빌드 abort 제거.
- [x] `dtcu_compare` 정확도 PASSED 확인 + perf baseline 기록 (`DTCU_baseline_dtcu_compare.md`).
- 남은 것: 없음. (dtcu_basic은 건드리지 않음.)

### Phase 1 — operand load를 helper로 분리 (동작 불변)
**목표:** 리팩터만. 결과/카운터 완전 동일.
- 추가/변경 (`d_tensor_core.cpp/.h`):
  - `build_op_req_lines_(uint32_t k_idx, std::vector<uint64_t>& out)` — 기존 `build_req_lists_()`의 operand(A/B/C) 부분을 k_idx 파라미터로 일반화.
  - `load_operands_into(uint32_t buf_idx, uint32_t k_idx)` — 기존 `load_operands()`를 (지금은 buf_idx=0 고정) k_idx 기준으로 일반화.
  - `calculate_base_A_/B_(k_idx)` 처럼 K 인덱스를 인자로 받게 정리 (현재는 멤버 `tile_k_idx_` 의존).
- FSM/버퍼 변경 없음 (buf_idx는 항상 0).
- **Acceptance:** `dtcu_compare` PASSED, cycles/instrs/MemReq 카운트 baseline과 동일.

### Phase 2 — ping-pong operand buffer 도입
**목표:** 버퍼만 2개로. 아직 overlap 없음(serial 유지).
- 변경:
  - `a_buf_` → `std::array<std::vector<uint32_t>,2>`, `b_buf_`도 동일. `accum_buf_`는 단일 유지(출력 tile 단위 누적).
  - 상태 필드: `uint32_t compute_buf_`, `uint32_t prefetch_buf_`, `uint32_t buf_k_idx_[2]`, `bool buf_ready_[2]`.
  - 동작: K마다 `load_operands_into(buf, k)` → `execute_mma()`(해당 buf 사용) → buf 토글. (여전히 load 후 compute)
- **Acceptance:** 두 버퍼 모두 사용됨(디버그 출력로 확인), `dtcu_compare` PASSED, 카운터 ~동일.

### Phase 3 — producer/consumer overlap (핵심)
**목표:** compute(K_k)와 prefetch(K_{k+1})를 동시에.
- 새 상태(또는 내부 sub-FSM):
  - `PREFETCH_REQ` / `PREFETCH_WAIT` (operand MemReq를 next buffer로 발행, 독립 인덱스 `tma_req_idx_`).
  - 응답 핸들러(`tick()` 상단)가 `pending_tag_`(main)와 `tma_pending_tag_`(prefetch) **둘 다** 매칭하도록 확장 → compute-phase와 prefetch가 독립 in-flight 가능.
- 코디네이션 플래그 (노트 요구: 분리 유지):
  - `compute_done_` (exec_cycles_left_==0), `tma_done_[buf]` (next buffer load 완료).
- 흐름:
  1. K0를 buf0로 blocking preload.
  2. for k: `exec_cycles_left_` 세팅하고 buf[cur] compute 시작; 동시에 k+1<tiles_k면 buf[next]로 prefetch 시작.
  3. 다음 K로 넘어가는 조건 = `compute_done_ && (다음 버퍼 tma_done_ || 다음 K 없음)`.
     - compute 끝났는데 next 버퍼 미완 → `dtcu_wait_for_tma_cycles_++`.
     - prefetch 끝났는데 버퍼가 아직 compute에 점유 → `tma_wait_for_buffer_cycles_++`.
  4. 마지막 K compute 후 OUT_REQ (store는 현행 유지).
- **Acceptance:** `dtcu_compare` **PASSED (숫자 출력 baseline과 동일)** + cycles가 baseline보다 감소 + 두 버퍼 사용.

### Phase 4 — counter 추가 & 측정
- 추가 counter: `tma_addr_gen_cycles_`, `tma_mem_wait_cycles_`, `tma_buffer_write_cycles_`, `dtcu_compute_cycles_`, `dtcu_wait_for_tma_cycles_`, `tma_wait_for_buffer_cycles_`.
- 노출: 우선 기존 `[DTCU] ...` 디버그 출력 라인에 추가 (CSR plumbing 없이). 필요 시 후속에서 MPM CSR 연결.
- **핵심 지표:** `dtcu_wait_for_tma_cycles_` — prefetch 전(=Phase2)/후(=Phase3) 비교해서 감소 입증.
- **Acceptance:** baseline 대비 표로 cycles, dtcu_wait_for_tma_cycles 비교 리포트.

### Phase 5 — output store overlap (보류, 후속)
- 완료된 D tile store와 다음 output tile의 첫 K load를 겹침.
- single TMA engine이라 load/store 경쟁 → arbitration 정책(Load 우선) 필요. Phase 4 안정화 후 착수.

---

## 3. 검증 방법 (매 Phase 공통)
- apptainer + tutorial 설정으로 `dtcu_compare` 실행, **PASSED 유지**(정확도 회귀 없음) 우선 확인.
- Phase 3/4에서 baseline(`DTCU_baseline_dtcu_compare.md`)과 cycles/MemReq/대기 counter 비교.
- 작은 사이즈(예: 64·32·16, 128·64·32)로 빠른 회귀 + 큰 사이즈로 overlap 효과 확인.

## 4. 리스크 / 메모
- outstanding 1개 모델에서 overlap 이득은 "compute cycle ≥ prefetch mem latency"일 때 큼. 작은 K-tile에선 이득이 작을 수 있음 → 측정으로 확인.
- `accum_buf_`는 ping-pong 대상 아님(출력 tile 단위 누적)이라 그대로.
- in-core TCU/DTCU의 `instr_tcu=0` 카운터 이슈는 본 계획과 별개(원하면 별도 Phase로).
- 각 Phase는 독립 커밋 + `dtcu_compare` PASSED 확인 후 다음 Phase 진행.
