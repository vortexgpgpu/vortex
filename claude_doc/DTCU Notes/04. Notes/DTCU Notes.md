
# 질문
* WMMA 설명하면서 `fragA_`, `fragB_`, `fragC_`를 lane별 fragment 배열로 들고있다고 했잖아. 여기서 말하는 lane은 워프 안에서 활성화된 쓰레드를 말하는거잖아. 근데 우리는 DTCU인데 이 방식이 잘못되지 않았나?
* TMA (DXA update set) 
* Asynchrnous Barrier
* Check with Shinnung

What is dispatcher

# Note

## 1. 지금 `fragA_`, `fragB_`, `fragC_`의 `lane`은 “실제 활성 warp thread”가 아닙니다

제가 앞서 `lane별 fragment 배열`이라고 설명한 건 **자료구조 모양**을 말한 것이고,  
여기서의 `lane`은 지금 DTCU 안에서는 **실제 실행 중인 SIMT lane**이 아닙니다.

왜냐하면 DTCU 내부는 아예 이렇게 잡혀 있기 때문입니다.

- `sim/simx/d_tensor_core.cpp:29`  
    `using cfg = vt::wmma_config_t<NUM_THREADS>;`
- `sim/simx/d_tensor_core.h:101-103`  
    `fragA_`, `fragB_`, `fragC_` 크기가 전부 `NUM_THREADS * ...`
- `sim/simx/d_tensor_core.cpp:244`, `284`, `325`, `808`, `843`, `858`  
    전부 `for (uint32_t lane = 0; lane < NUM_THREADS; ++lane)`로 돕니다

즉, 이 `lane`은 **warp scheduler가 관리하는 live lane**이 아니라,  
**WMMA tile을 software적으로 쪼개기 위한 index**로만 쓰이고 있습니다.

실제로 DTCU kernel은 leader만 실행합니다.

- `tests/regression/dtcu_basic/kernel.cpp:10-15`
- `tests/regression/dtcu_compare/kernel.cpp:50-57`

둘 다 `if (tid == 0 && wid == 0)`에서만 `dtensor_start()`와 `dtensor_poll()`를 부릅니다.  
즉 DTCU compute에 실제 warp 전체가 참여하는 구조가 아닙니다.

---

## 2. 그래서 “기능적으로는 맞을 수 있지만, DTCU 모델링으로는 어색하다”가 정확한 표현입니다

현재 구현은 `load_operands()`가 아예 주석으로도 적어놓았듯이  
**`kernel/include/vx_tensor.h`의 WMMA fragment mapping을 그대로 따라갑니다**.

- `sim/simx/d_tensor_core.cpp:243`  
    `// Load A (row_major), same mapping as kernel/include/vx_tensor.h`
- `sim/simx/d_tensor_core.cpp:283`  
    `// Load B (col_major)`
- `kernel/include/vx_tensor.h:176-272`  
    lane별 `load_matrix_sync()` mapping
- `sim/simx/d_tensor_core.cpp:821-828`  
    WMMA execution path를 그대로 재사용
- `sim/simx/d_tensor_core.cpp:349`  
    `// Start of FMA and FEDP definitions (copied from tensor_unit.cpp)`

즉 지금 DTCU는

1. A/B/C를 **WMMA lane-fragment 배치**로 `fragA_/fragB_/fragC_`에 넣고
2. `execute_wmma()`에서 그걸 lane index 기준으로 다시 꺼내
3. 기존 TensorUnit의 FMA/FEDP 로직을 써서 계산하고
4. 결과를 같은 mapping으로 D에 저장합니다

그래서 **수학 결과를 in-core TCU와 일치시키는 데에는 유리**합니다.  
하지만 구조적으로는 말씀하신 대로 **DTCU가 warp/lane abstraction에서 독립적이지 않습니다.**

---

## 3. 왜 이게 DTCU 관점에서 문제처럼 보이느냐

이유는 분명합니다.

### (a) tile shape가 `NUM_THREADS`에 묶여 있습니다

`wmma_config_t`는 아예 `NT`를 받아 tile shape를 결정합니다.

- `sim/common/tensor_cfg.h:149-220`

즉 현재 DTCU의

- `tileM`
- `tileN`
- `tileK`
- `NRA/NRB/NRC`

가 전부 `NUM_THREADS`에 의해 정해집니다.

이건 **warp-level WMMA**에는 자연스럽지만,  
**cluster-level decoupled accelerator**라면 원칙적으로 부자연스럽습니다.  
DTCU라면 tile shape는 accelerator 자체의 microarchitecture 파라미터여야지,  
core의 warp width에 종속되면 안 되는 게 맞습니다.

---

### (b) accumulator/state가 “lane-partitioned register fragment” 꼴입니다

`fragC_`는 본질적으로 `NUM_THREADS * cfg::NRC` 배열입니다.  
이건 결국 **“각 lane이 accumulator fragment 일부를 가진다”**는 WMMA mental model입니다.

DTCU라면 더 자연스러운 내부 표현은 예를 들면 이런 것들입니다.

- `tileM x tileK` 크기의 A tile buffer
- `tileK x tileN` 크기의 B tile buffer
- `tileM x tileN` 크기의 accumulator tile
- 혹은 PE 좌표 기반 buffer / systolic state

지금처럼 `lane * NRC + r` 식으로 접근하는 건 (`sim/simx/d_tensor_core.cpp:811, 847, 873`)  
명백히 WMMA register-fragment 관성을 끌고 온 겁니다.

---

### (c) 실제 DTCU 계산에는 “active lane” 개념이 전혀 안 쓰입니다

진짜 DTCU라면 최소한

- active warp mask
- thread mask
- warp participation

같은 SIMT 개념과 분리돼 있어야 합니다.

그런데 `execute_wmma()`에는 그런 게 없습니다.

- `sim/simx/d_tensor_core.cpp:796-799`  
    `rs1_data`, `rs2_data`, `rs3_data`, `rd_data`를 `NUM_THREADS` 크기로 고정 생성
- `sim/simx/d_tensor_core.cpp:808-818`  
    lane별로 값 pack
- `sim/simx/d_tensor_core.cpp:843-847`  
    lane별로 결과 write-back

즉 여기서 lane은 그냥 **고정된 software index**입니다.

---

## 4. 그러면 “잘못된 구현”이냐?

이건 목적에 따라 답이 달라집니다.

### 기능 검증 목적이라면

완전히 잘못됐다고 할 수는 없습니다.

왜냐하면 지금 구현 목표가 사실상

- **기존 WMMA와 같은 semantics**
- **같은 tile decomposition**
- **같은 수치 결과**
- **start/poll + memory-based offload 경로 검증**

이라면, WMMA fragment mapping을 재사용하는 게 가장 빠르고 안전한 방법이기 때문입니다.

실제로 `dtcu_compare`가 바로 그걸 검증합니다.  
같은 수학 연산을 in-core TCU와 DTCU가 같은 결과로 내는지 비교하는 테스트입니다.

---

### 하지만 “진짜 DTCU/virgo-style decoupled unit”을 모델링하려는 목적이라면

네, **개념적으로는 잘못됐다고 보는 게 맞습니다.**

정확히 말하면:

- **functional shortcut으로는 허용 가능**
- **architectural model로는 부정확**

입니다.

특히 아래 두 점이 가장 큽니다.

1. **DTCU tile shape가 warp width에 종속됨**
2. **내부 상태 표현이 PE/accumulator tile이 아니라 lane-fragment 기반임**

이 둘 때문에 현재 DTCU는 “core와 분리된 accelerator”라기보다  
**“warp-level WMMA engine을 core 밖으로 옮겨서 start/poll로 감싼 버전”**에 더 가깝습니다.

---




# 해야할 것


