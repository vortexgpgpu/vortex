# DTCU Perf Knobs (성능 스윕 노브 총정리)

> DTCU/TMA 타이밍 모델에서 **우리가 조정해 성능을 바꿀 수 있는 값** 전수 정리. bank 수·버퍼 크기·MAC/cycle 같은 것들이 여기 다 있다. 각 노브가 **어느 파일**에 있고, **무엇을** 정하고, **어느 계산**에 들어가고, **어떤 [[DTCU Perf Stat|MPM 카운터]]/성능 구간**에 영향을 주는지 표로 요약한다.
> 기준 소스: [dtcu_params.h](sim/simx/dtcu/dtcu_params.h) (v3.0, 2026-07). 관련: [[DTCU Latency Modeling]], [[DTCU Implementation]], [[DTCU Perf Stat]].

---

## 0. 핵심 3줄 요약

- **거의 모든 성능 노브는 [sim/simx/dtcu/dtcu_params.h](sim/simx/dtcu/dtcu_params.h) 한 파일**에 있고, 전부 `#ifndef` 가드라 **소스 수정 없이 빌드 시 `-D<이름>=<값>`으로 오버라이드**된다.
- 노브는 크게 **①compute 처리량(MAC/acc bank)**, **②operand SRAM(bank/swizzle)**, **③메모리 동시성(outstanding)**, **④타일 geometry(버퍼 크기)**, **⑤고정 latency** 다섯 갈래.
- compute 사이클은 **3-stage 파이프라인의 최댓값** 구조 → 어떤 노브가 먹히는지는 "지금 뭐가 bound냐"에 달림 (§4).

---

## 1. 빌드-오버라이드 가능 노브 (dtcu_params.h) — **메인 스윕 대상**

전부 `-D<이름>=<값>`로 재빌드 시 반영. 기본값은 소스 그대로.

| # | 노브 | 위치 | 기본값 | 무엇을 조정 | 사용처(코드) | 영향 카운터 / 성능 구간 |
|---|---|---|---|---|---|---|
| 1 | `DTCU_MACS_PER_CYCLE` | [dtcu_params.h:34](sim/simx/dtcu/dtcu_params.h#L34) | `16` | 매트릭스 어레이의 **MAC/cycle 처리량**(어레이 폭) | `mac_cycles = ceil(m·n·k / this)` — [dtcu.cpp:329](sim/simx/dtcu/dtcu.cpp#L329) | `compute`. **MAC-bound**일 때만 먹힘. 올리면 physically wider array 모델 → compute↓ (단 acc bank가 floor를 잡으면 무효, §4) |
| 2 | `DTCU_ACC_BANKS` | [dtcu_params.h:58](sim/simx/dtcu/dtcu_params.h#L58) | `2` | **누산기 SRAM 대역폭**(fp32 word/cycle, 충돌 없음) | ①compute accum stage `ceil(2·m·n/this)` [dtcu.cpp:332](sim/simx/dtcu/dtcu.cpp#L332) ②store drain accread [dtcu_tma.cpp:513](sim/simx/dtcu/dtcu_tma.cpp#L513) ③C-load fill [dtcu_tma.cpp:420](sim/simx/dtcu/dtcu_tma.cpp#L420) | `compute`(accum-bound floor), `store_drain`/`store_wait`, `buf_write`. **큰 타일에선 이게 compute 바닥**을 결정(§4) |
| 3 | `DTCU_SMEM_BANKS` | [dtcu_params.h:98](sim/simx/dtcu/dtcu_params.h#L98) | `2` | **operand(A/B) 스크래치패드 bank 수**(1 word/bank/cycle). 2의 거듭제곱 | ①operand read 충돌 히스토그램 [dtcu.cpp:284–305](sim/simx/dtcu/dtcu.cpp#L284-L305) ②TMA fill rate `ceil(op_words/this)` [dtcu_tma.cpp:415](sim/simx/dtcu/dtcu_tma.cpp#L415) ③`bank_of_()` mask [dtcu.cpp:275](sim/simx/dtcu/dtcu.cpp#L275) | `opread`, `buf_write`. **B 컬럼 읽기가 bound**일 때 먹힘. `DTCU_SWIZZLE`와 짝 |
| 4 | `DTCU_SWIZZLE` | [dtcu_params.h:107](sim/simx/dtcu/dtcu_params.h#L107) | `0` | operand SRAM **bank swizzle**(0=naive, 1=XOR-permute) | `bank_of_()` — [dtcu.cpp:271–276](sim/simx/dtcu/dtcu.cpp#L271-L276) | `opread`. B 컬럼(stride `TILE_N_MAX`)이 한 bank에 aliasing→충돌. `1`이면 bank 분산. **banks>1일 때만 의미**. 기능값 불변, 타이밍만 |
| 5 | `DTCU_MAX_OUTSTANDING` | [dtcu_params.h:78](sim/simx/dtcu/dtcu_params.h#L78) | `=VX_CFG_L2_MSHR_SIZE`(16) | **동시 in-flight prefetch/store 요청 수**(memory-level parallelism) | FETCH 발행 가드 `inflight < this` [dtcu_tma.cpp:447](sim/simx/dtcu/dtcu_tma.cpp#L447), [482](sim/simx/dtcu/dtcu_tma.cpp#L482) | `mem_wait`. **memory-bound**일 때 올리면 동시 L2 miss↑ → mem_wait↓. L2 MSHR에 물려있음(그 이상은 L2가 막음) |
| 6 | `DTCU_ADDRGEN_CYCLES` | [dtcu_params.h:71](sim/simx/dtcu/dtcu_params.h#L71) | `3` | **AGU 주소생성 per-K-tile setup** 사이클 | `tma_addrgen_left_ = this` (ADDRGEN state) [dtcu_tma.cpp:407](sim/simx/dtcu/dtcu_tma.cpp#L407) | `addrgen`. K 타일마다 고정 setup. overlap으로 대부분 숨겨짐 |
| 7 | `DTCU_BUF_LATENCY` | [dtcu_params.h:46](sim/simx/dtcu/dtcu_params.h#L46) | `1` | operand 스크래치패드 **기본 접근 latency**(L1 dcache 모델) | operand read `+this` [dtcu.cpp:330](sim/simx/dtcu/dtcu.cpp#L330), fill `+this` [dtcu_tma.cpp:422](sim/simx/dtcu/dtcu_tma.cpp#L422) | `opread`(read stage), `buf_write`. 고정 오프셋 |
| 8 | `DTCU_ACC_LATENCY` | [dtcu_params.h:61](sim/simx/dtcu/dtcu_params.h#L61) | `=DTCU_BUF_LATENCY`(1) | 누산기 SRAM **기본 접근 latency** | accum stage `+this` [dtcu.cpp:332](sim/simx/dtcu/dtcu.cpp#L332) | `compute`(accum stage 상수항) |
| 9 | `DTCU_COMPUTE_LATENCY` | [dtcu_params.h:37](sim/simx/dtcu/dtcu_params.h#L37) | `6` | native 타일당 **파이프라인 fill latency** | `compute = max(...) + this` [dtcu.cpp:334](sim/simx/dtcu/dtcu.cpp#L334) | `compute`. K 타일마다 고정 가산 |
| 10 | `DTCU_TILE_M` | [dtcu_params.h:86](sim/simx/dtcu/dtcu_params.h#L86) | `64` | native 타일 **M**(고정). A/누산기 버퍼 크기·per-tile 일량 | 버퍼 assign [dtcu.cpp:227–232](sim/simx/dtcu/dtcu.cpp#L227-L232), `tile_m_` | 전 카운터(일량 스케일). **버퍼 크기 = TILE_M×…** |
| 11 | `DTCU_TILE_N_MAX` | [dtcu_params.h:89](sim/simx/dtcu/dtcu_params.h#L89) | `128` | **최대** native 타일 N = **operand/acc SRAM 물리 용량**. B 컬럼 물리 stride | 버퍼 assign [dtcu.cpp:229–232](sim/simx/dtcu/dtcu.cpp#L229-L232), B stride [dtcu.cpp:640](sim/simx/dtcu/dtcu.cpp#L640)/[dtcu_tma.cpp:278](sim/simx/dtcu/dtcu_tma.cpp#L278), swizzle shift [dtcu.cpp:273](sim/simx/dtcu/dtcu.cpp#L273) | **버퍼(SRAM) 용량 노브**. 실제 tile_n은 desc의 `shape_n_size·16`로 이 이하에서 선택(§3) |

> **주의(power-of-two)**: `DTCU_SMEM_BANKS`는 `& (banks-1)` 마스크·swizzle 시프트를 쓰므로 **2의 거듭제곱**이어야 한다([dtcu.cpp:275](sim/simx/dtcu/dtcu.cpp#L275)). `DTCU_ACC_BANKS`는 `ceil` 나눗셈뿐이라 임의 값 가능하지만 관례상 2의 거듭제곱 권장.

---

## 2. 고정 geometry 상수 (⚠️ `-D` 오버라이드 **불가** — 소스 수정 필요)

`#ifndef` 가드가 없는 `constexpr`. 바꾸려면 **두 파일 다** 고쳐야 한다(중복 정의).

| 상수 | 위치(2곳) | 값 | 무엇 | 바꿀 때 주의 |
|---|---|---|---|---|
| `DTCU_TILE_K_WORDS` | [dtcu.cpp:36](sim/simx/dtcu/dtcu.cpp#L36) **&** [dtcu_tma.cpp:30](sim/simx/dtcu/dtcu_tma.cpp#L30) | `8` | native 타일 **K 깊이(패킹된 32b word 수)** = 32 B/K-tile (fp16 16개 / fp32 8개) | 두 정의를 **동시에** 수정해야 함. A/B 버퍼의 K 차원·operand read 히스토그램·fill word 수에 전부 물림. `-D`로 못 바꾸니 스윕 대상이면 상수 → `#ifndef`화 먼저 |

> 스윕하고 싶으면 `DTCU_TILE_K_WORDS`를 `dtcu_params.h`로 옮겨 `#ifndef` 가드로 만드는 소규모 리팩터를 먼저 하는 걸 권장(현재는 하드코딩).

---

## 3. Config에서 상속되는 노브 (VX_config, `-DVX_CFG_*`)

DTCU 전용은 아니지만 DTCU 타이밍에 직접 영향. TOML→헤더 파이프라인(`ci/gen_config.py`)이 방출.

| 노브 | 위치 | 기본 | DTCU에 미치는 영향 |
|---|---|---|---|
| `VX_CFG_L2_MSHR_SIZE` | [VX_config.h:860](sw/VX_config.h#L860) | `16` | `DTCU_MAX_OUTSTANDING` 기본값을 **그대로 물려줌** → prefetch 동시성 상한. `mem_wait` |
| `VX_CFG_XLEN`(÷8) | (config) | 32b→`4B` | 스크래치패드 word 크기(1 word/bank/cycle의 word 단위) |
| L2 뱅크/포트/latency 계열 (`VX_CFG_L2_*`) | VX_config | — | DTCU는 **L2 arbiter 행 1개 공유** → L2 설정이 곧 DTCU 메모리 응답 시간. `mem_wait` (엔진↔코어 L2 경쟁, [[DTCU Perf Stat]] §2 교차검증) |

---

## 4. 어느 노브가 먹히나 — compute 파이프라인 구조 (⭐ 스윕 전 필독)

한 K 타일 compute 비용([dtcu.cpp:308–335](sim/simx/dtcu/dtcu.cpp#L308-L335))은 **세 스테이지의 합이 아니라 최댓값** + 고정 latency:

```
compute(K타일) = max( mac_cycles, read_cycles, accum_cycles ) + DTCU_COMPUTE_LATENCY

  mac_cycles   = ceil( tile_m · tile_n · tile_k / DTCU_MACS_PER_CYCLE )
  read_cycles  = operand_read_cycles_()            + DTCU_BUF_LATENCY   // SMEM_BANKS·SWIZZLE 의존
  accum_cycles = ceil( 2 · tile_m · tile_n / DTCU_ACC_BANKS ) + DTCU_ACC_LATENCY
```

**함의 — 노브를 바꿔도 다른 스테이지가 bound면 효과 0:**
- **`DTCU_MACS_PER_CYCLE`를 아무리 올려도** `accum_cycles`(누산기 대역폭)가 더 크면 compute는 안 줄어든다. 예: 2048-element 타일은 `ACC_BANKS=2`에서 `ceil(2·2048/2)=2048` 사이클이 **바닥** → MAC을 무한대로 해도 여기서 안 내려감([dtcu.cpp:320–323](sim/simx/dtcu/dtcu.cpp#L320-L323) 주석의 "real bandwidth limit").
  - → MAC을 넓혔으면 **`DTCU_ACC_BANKS`도 같이 올려야** 실제 이득.
- **`DTCU_SMEM_BANKS`/`DTCU_SWIZZLE`는 `read_cycles`가 bound일 때만** 체감. B 컬럼 충돌이 심한 형상에서 `SWIZZLE=1` 또는 bank↑가 `opread`를 깎음.
- **`DTCU_MAX_OUTSTANDING`은 memory-bound(`mem_wait` 큼)일 때만** 체감. compute-bound면 무효.

**진단 순서**(= [[DTCU Perf Stat]] 카운터 읽고 병목부터):
1. `wait_tma` 큼 → **memory-bound** → `DTCU_MAX_OUTSTANDING`↑, L2 설정, `mem_wait` 확인.
2. `compute` 큰데 `mac`이 아니라 accum이 floor → `DTCU_ACC_BANKS`↑.
3. `opread` 큼 → `DTCU_SWIZZLE=1` / `DTCU_SMEM_BANKS`↑.
4. `store_drain` 큼(마지막 타일 store 안 숨겨짐) → `DTCU_ACC_BANKS`↑(accread rate) / 형상 조정.

---

## 5. 런타임(소프트웨어) 노브 — 재빌드 없이 desc/테스트로

빌드 노브는 아니지만 성능을 바꾸는 값. 참고로 표에 포함.

| 노브 | 어디서 | 무엇 | 영향 |
|---|---|---|---|
| `shape_n_size` (descriptor) | 커널 desc → `tile_n_ = shape_n_size·16` (≤ `DTCU_TILE_N_MAX`) | 실제 native 타일 **N 폭** | 타일당 일량·`accum_cycles` floor·B 컬럼 충돌 정도 |
| GEMM 크기 M/N/K | 테스트 `-DSIZE_MULT=N` ([dtcu_compare](tests/regression/dtcu_compare/main.cpp#L25)) | 전체 GEMM = SIZE_MULT × native 타일 | 타일 수(`tiles_m/n/k`) → overlap 관측 구간·store drain 비중 |

---

## 6. 스윕 실전 (빌드 예시)

노브는 SimX **컴파일 define**이라 테스트 `CONFIGS`에 `-D`로 얹으면 된다(게이트 플래그 `-DVX_CFG_EXT_DTCU_ENABLE`와 함께).

```bash
# 예: 누산기 대역폭 2배 + operand swizzle on 으로 dtcu_compare 스윕
make -C tests/regression/dtcu_compare \
  CONFIGS="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DTCU_ENABLE \
           -DDTCU_ACC_BANKS=4 -DDTCU_SWIZZLE=1 -DDTCU_MACS_PER_CYCLE=32" run-simx
```

측정은 [[DTCU Perf Stat]]의 explicit `vx_mpm_query`(**클래스 9**) 경로로 11개 카운터를 읽어 비교한다(`dtcu_compare`가 자동 출력하는 `[DTCU MPM]`). 노브를 바꾼 뒤 **어느 카운터가 움직였나**로 효과를 검증할 것.

> ⚠️ 노브를 바꾸면 **기능 정확성은 그대로**(functional `execute_mma()`는 값 oracle, 타이밍 모델만 바뀜 — [dtcu.cpp:326](sim/simx/dtcu/dtcu.cpp#L326) 주석)여야 한다. `dtcu_compare`가 여전히 PASS(TCU==DTCU==CPU ref)인지 먼저 확인하고 사이클을 논할 것.
