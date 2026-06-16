---
share_link: https://share.note.sx/drf5ruk9#MHh8eGA1Xe/t/muIU/n6xYrNbJIO7IW7/YuwuhPlsO0
share_updated: 2026-04-08T13:44:15-04:00
---


목차

- [[#Backgrounds|Backgrounds]]
- [[#Direction|Direction]]
- [[#Implementation 방향|Implementation 방향]]
- [[#1차 구현 목표|1차 구현 목표]]
- [[#2차 확장 목표|2차 확장 목표]]


---
# Backgrounds

## Introduction
> Virgo/Hopper/Blackwell 모두 thread 수에 따라 Tile Size를 동적으로 바꾸지 않음.
> Hardware가 여러개의 Tile Size를 준비하면 소프트웨어가 이 중에서 선택해서 kernel tile을 정함.
> Virgo는 선택 폭이 거의 없고, Hopper는 instruction mnemonic의 `.shape`으로, Blackwell은 `tcgen05.mma`의 `idesc`와 `cta_group/.ws/kind` 조합으로 Tile Size가 정해짐.

### Tile Size 정의
1. **Instruction/native tile**: 하드웨어 MMA/MMU가 한 번에 처리하는 최소 연산 단위
		-> _내가 아는 Tile Size의 정의
2. **Kernel/block tile**: 커널이 shared memory, thread block, warpgroup, CTA group을 이용해 위 native tile들을 묶어서 쓰는 상위 타일

---
## Virgo
> DTCU의 hardware configuration이 Native Tile Size를 정함
> 논문 속 Configuration은 128 × 64 × 128
> 타일 크기에 맞춰 thread block이 협업함

### Deciding Tile Size
* SIMT thread 수가 native tile을 정하는 것이 아님
* 타일이 **thread block size를 결정**
	* Vortex랑 반대
	* **tile → thread block mapping**이지, **thread block → tile 결정**이 아님

> **4.4.1 Thread Block Tiling**
> In our configuration, the matrix unit exposes 128×64×128 as the tile size of a single operation, which determines the thread block size. 
> 
> Each thread block, spatially partitioned across the (M,N) output space, completes the full GEMM by accumulating across the K dimension temporally in a loop, shown in Figure 4. 
> 
> As the loop iterates, the Gemmini matrix unit accumulates partial sum data onto its _private accumulator memory_, which gets moved out and stored to the global memory at the end of the loop. 
> 
> Then, the kernel moves on to the next (M,N) output tile that is allocated to that thread block. The kernel can launch multiple thread blocks, where the (M,N) output space is divided equally across the thread blocks.



### 큰 GEMM 처리 방식
> 더 큰 GEMM은 이 native tile을 기준으로 **K dimension temporal accumulation** (fragC)와 `(M,N)` space tiling으로 확장

<타일보다 큰 GEMM이라면?>
* 내부에 FSM이 있어서 **`i, j, k`를 자동으로 iterate**
* Kernel 쪽에서는 DMA, barrier, fence로 상위-level tiling을 맞춤
* 즉 Virgo는 native tile 하나를 크게 잡고, **더 큰 GEMM은 native tile 반복으로 덮는 구조**


#virgo #tile_size 

---
## Hopper
> Native Tile => `M=64` 고정; `N`은 선택; `K`는 datatype에 따라 결정됨 (8/16/32/256)
> 하드웨어가 `.shape` set를 정하고 (가능한 Tile Size), 소프트웨어가 이 중 선택
> Warp group 전체가 같은 instruction을 issue해야 하지만, thread 수가 tile을 runtime에 늘리거나 줄이지는 않음


`wgmma.mma_async`는 **warpgroup-level instruction**

PTX는 `.aligned` qualifier에 대해 **warpgroup의 모든 thread가 같은 `wgmma.mma_async`를 실행해야 함**

따라서 Hopper는 “active thread 수에 따라 tile이 바뀌는 구조”가 아니라, **warpgroup이라는 fixed execution granularity 위에서 software가 legal `.shape`를 선택하는 구조**

### Deciding Tile Size
* Hopper의 Warp-Group MMA의 Tile Size는 **M이 64 고정**임.
	* 즉, Native Tile Size 옵션들이 전부 `m64n?k?`임
* N과 K값은 "Legal Set" 안에서 datatype에 따라 달라짐.

- **FP16 / BF16**
    - `m64n8k16`부터 `m64n256k16`까지
    - **N은 8 단위 증가, K는 16**
- **TF32**
    - `m64n8k8`부터 `m64n256k8`까지
    - **N은 8 단위 증가, K는 8**
- **FP8 (`e4m3`, `e5m2`)**
    - `m64n8k32`부터 `m64n256k32`까지
    - **N은 8 단위 증가 , K는 32**
- **INT8 / UINT8**
    - `m64n8k32, m64n16k32, m64n24k32, m64n32k32, m64n48k32, ... , m64n224k32`
    - **N이 8 단위 증가가 아니라, PTX가 정한 이산 집합; K는 32 고정**
- **1-bit**
    - `m64n8k256`부터 `m64n256k256`까지
    - **K가 256으로 커짐**



## Blackwell

https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm100.html 


---
# Direction
> 현재 우리 DTCU는 Tile Size가 thread structure에 종속된 WMMA-derived design. Hopper/Blackwell/Virgo는 Tile Size가 Explicit hardware-supported shape인 design.
> 따라서 우리는 Tile Size를 thread count에서 분리해야함.

**1. tile shape를 thread count에서 분리**
DTCU native tile은 `NUM_THREADS`에서 계산하지 말고, 별도 hardware parameter둠

예를 들어,
- `DTCU_TILE_M`
- `DTCU_TILE_N`
- `DTCU_TILE_K`
ㅊ처럼개를 독립 config로 두고, descriptor의 `M/N/K`는 **문제 크기(problem size)** 로만 쓰고, native tile은 저 고정 parameter를 기준으로 내부 loop를 돌게 해야됨


**2. 내부 storage를 lane-fragment 구조에서 tile buffer / accumulator buffer 구조로 변경**
`fragA_[lane * ...]`, `fragB_[lane * ...]`, `fragC_[lane * ...]` 구조를 유지하면, tile이 커질수록 indexing도 복잡해지고 본질적으로 “lane 수가 tile shape를 규정하는 구조”를 못 벗어남.


**3. target tile은 NVIDIA 숫자를 그대로 따라가기보다, 메모리 footprint로 정해야됨**  
fp16 input / fp32 accumulate 기준으로 보면:
- `128x64x128`
    - A = 128×128×2B = **32KB**
    - B = 128×64×2B = **16KB**
    - D/C = 128×64×4B = **32KB**
    - 총 핵심 tile footprint가 대략 **80KB**
- `64x256x16`
    - A = **2KB**
    - B = **8KB**
    - D/C = **64KB**
    - 총 **74KB**
- `256x256x16`
    - A = **8KB**
    - B = **8KB**
    - D/C = **256KB**
    - 총 **272KB**

현재 상황을 고려 (dedicated accumulator SRAM/DMA없음) 고려해서 현실적인 크기 조정


---
# Implementation 방향
1. **native tile shape를 `NUM_THREADS`에서 완전히 분리**
2. **내부 storage를 per-lane `fragA_/fragB_/fragC_`에서 dense tile / accumulator buffer로 변경**

## Thread Dependency 해소 방안 
### (1) tile size 자체
`sim/common/tensor_cfg.h`

- `tile_cap = NT * NR`
- `xtileM`, `xtileN`, `xtileK`
- 최종 `tileM`, `tileN`, `tileK`

### (2) DTCU의 native tile 계산
`sim/simx/d_tensor_core.cpp`

- `tile_k_elems = cfg::tileK * i_ratio`
- `M/N/K`가 `cfg::tileM/cfg::tileN/tile_k_elems`의 배수인지 검사
- `tiles_m_`, `tiles_n_`, `tiles_k_` 계산

즉 DTCU outer loop도 여전히 `cfg::tileM/tileN/tileK`에 묶여 있음.

### (3) 내부 load/compute/store
- `load_operands()` : `sim/simx/d_tensor_core.cpp:233-347`
- `execute_wmma()` : `sim/simx/d_tensor_core.cpp:791-852`
- `store_output()` : `sim/simx/d_tensor_core.cpp:854-877`
- `build_req_lists_()` : `sim/simx/d_tensor_core.cpp:898-1023`

여기는 전부 lane별로 tile을 잘라 담고, lane별로 읽고, lane별로 계산하고, lane별로 저장

> 따라서 이쪽에서 frag iterate하고 연산하는 구조를 다 바꿔야됨!!!!!

---
## Tile Shape 알고리즘

### 새 DTCU native tile rule
> Hopper 방식을 참고

- `M = 64` 고정
- `N`은 legal set에서 software가 선택
- `K`는 input datatype에 따라 결정
    - fp16 input이면 `K = 16`
    - fp32 input이면 `K = 8`

이미 구현된 `i_ratio` logic을 그대로 재활용 (필요한 k 타일 갯수 구할 때) 
> `uint32_t i_ratio = 4 / in_sz;`

- 내부 기준 `tileK_words = 8` 로 둠
	- tileK_words는 내부 표현
		- `word = 4 bytes`
		- `tileK_words = 8`이면
		- **K dimension의 operand byte 폭을 8개의 4-byte word로 본다**는 뜻
		- 쉽게 말해 Hopper처럼 fp16을 K=16, fp32를 K=8이 될 수 있도록 바꿔주는 coefficient
- `tileK_elems = tileK_words * (4 / elem_size(fmt_s))`

그렇게 되면 K값은
- fp16 input: `8 * (4 / 2) = 16`
- fp32 input: `8 * (4 / 4) = 8`


이렇게 하는 이유: fp16이든 fp32든 **A/B operand byte footprint가 같아짐.

| A Tile Bytes                                                                                         | B Tile Bytes                                                                                   |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `A = M * K * sizeof(input)`<br>fp16: `64 * 16 * 2 = 2048B = 2KB`<br>fp32: `64 * 8 * 4 = 2048B = 2KB` | `B = K * N * sizeof(input)`<br>fp16: `16 * N * 2 = 32N bytes`<br>fp32: `8 * N * 4 = 32N bytes` |

즉 input type과 무관하게 A/B 값이 고정되어 data type이 바뀌어도 operand tile의 byte footprint가 같아짐
- **A = 2KB** 고정
- **B = 32N bytes** 고정

Hopper도 dtype에 따라 K를 바꿔서 이거랑 똑같이 함

---
## Accumulator Buffer 

### Tile Size 구하는 법
fp16 input / fp32 accumulate일 때:
- A = 2KB
- B = 8KB
- Accumulator(C/D tile) = 64 x 256 일때
	- 크기는 `64 * 256 * 4Byte = 64KB`
- 총 = **74KB**

그런데 여기서 진짜 부담은 A/B가 아니라 **accumulator**

수식으로 쓰면, `M=64`, `K_words=8`, internal accumulation fp32일 때 총 resident footprint는:
- `A = 2048B`
- `B = 32N`
- `Acc = 256N`

즉 총합은
- **`Total = 2048 + 288N` bytes**
	- 각 N값에 대해서:
		- `N=64` → `2048 + 18432 = 20480B` ≈ **20KB**
		- `N=128` → `2048 + 36864 = 38912B` ≈ **38KB**
		- `N=192` → `2048 + 55296 = 57344B` ≈ **56KB**
		- `N=256` → `2048 + 73728 = 75776B` ≈ **74KB**

---
# 1차 구현 목표

### 최대 Tile Shape 크기
* `64 x 128 x 16` (fp16) / `64 x 128 x 8` (fp32)을 최대 shape 크기로 둠
	- 이렇게 하면 accumulator가 정확히 **32KB**
		- Virgo 논문도 **Accumulator Memory가 32KB**

최대 N인 128이라고 했을 때, output(accumulator) tile 크기 M × N = 64 × 128 = 8192 elements

어차피 accumulator data type은 fp32(=4 Byte)니깐 
```
8192 × 4 = 32768 Bytes
32768 Bytes = 32 KB 
```

### Tile Shape Set
Hardware적으로는 최대 128까지 지원하도록. 이후 16 배수로 늘어남.
* `{16, 32, 64, 128}` 또는 `{16, 32, 64, 96, 128}`
* 16 배수인 이유는 software 때문
	* fp32 accumulate일 때 D tile row width는 `4 * N` Byte
		- `N=16`이면 64B
		- `N=32`이면 128B
		- `N=64`이면 256B
		- `N=128`이면 512B
	- 즉 **N이 16의 배수이면 row width가 64B cache line에 정확히 정렬됨

따라서,
- `M = 64`
- `N ∈ {8, 16, 24, ..., 128}`
- `K = 8 * (4 / elem_size(fmt_s))`
    - fp16 → 16
    - fp32 → 8


> Hopper PTX의 `wgmma`도 FP16/BF16에서 `m64n8k16`부터 `m64n256k16`까지 legal `.shape`를 두고, warpgroup 전체가 같은 instruction을 실행함
> 또 CUTLASS는 이 legal instruction shapes 중 무엇을 kernel generation에 포함할지 software/library가 고르는 구조

### Software가 Legal Set 중 고르는 방법
Hopper를 따라가려면 software가 tile shape을 고를 수 있어야됨.

다만 현재 Vortex 환경은 `dtensor_start(desc_addr)`로 전체 GEMM을 DTCU가 내부 FSM으로 처리함. 따라서 descriptor를 통해서 소프트웨어가 tile shape을 정할 수 있도록 구성.

`dtensor_desc_t` 중 reserved field를 shape selector로 사용.
- `reserved0` → `shape_n_code`
    - `0` = AUTO
    - `1` = `N=8`
    - `2` = `N=16`
    - ...
    - `32` = `N=256`
- `reserved1` → `shape_policy`
    - `0` = exact
    - `1` = auto_preferred
    - `2` = auto_largest_fit

이렇게 하면 descriptor size 64B도 그대로 유지됨.

이렇게 Descriptor를 바꾸면 hardware (`init_tile_state_()`) 에선:
- `tileM_ = 64`
- `tileK_words_ = 8`
- `tileK_ = tileK_words_ * (4 / in_sz)`
- `tileN_ = decode_shape_n(desc_.shape_n_code, desc_.shape_policy, problemN, accum_capacity)`
처럼 정해져서 runtime에 tile shape이 정확히 정해짐

따라서 host에서 desriptor값도 정확히 기재해야됨. (dtcu_basic 바꾸기)
- `shape_n_code = 16` → `N=128`
- `shape_policy = exact`

### 내부 storage 방식 변경 (frag -> tile buffer)
현재는 내부 storage가 frag로 저장됨 
* `fragA_`, `fragB_`, `fragC_`는 WMMA-style fragment임
	- tile 전체를 한 덩어리로 저장하는 게 아니라
	- lane별로 쪼개진 조각
	- 각 lane/thread가 자기 몫의 register 조각을 들고 있는 형태

즉 현재 `frag`는  lane count와 강하게 연결됨
- tile size를 바꾸면 lane mapping도 같이 다시 생각해야 함
- **thread와 tile shape의 분리**와는 반대

따라서 C/D는 accumulator buffer로 바꾸고, **A/B는 dense tile buffer로 바꾸는 것**이 타당 
([[DTCU Basic Theory#Dense Tile Buffer / Accumulator Buffer]] 참고)
([[DTCU Virgo#내부 Storage 방식]] 참고)

- `a_tile_` : `tileM x tileK`
- `b_tile_` : `tileK x tileN`
- `accum_buf_` : `tileM x tileN`


**A/B operand tile buffer 구현 방식** (32-bit word buffer)
```
std::vector<uint32_t> a_tile_words_;  
std::vector<uint32_t> b_tile_words_;
```

`uint32_t`의 벡터로 두면 버퍼의 index 1개는 matrix element 1개가 아니게 됨
* fp32 input이면 element 하나 당 **fp32 1개**
- fp16/bf16 input이면 element 하나 당 **16-bit 값 2개를 pack
- 이렇게 한 이유 
	1. `load_operand()` 에서도 operand를 logical element가 아니라 32-bit packed word로 받음
	2. `FEDP`도 32-bit word로 받음
```
auto a = reinterpret_cast<const itype *>(&a_row[z].u32);  
auto b = reinterpret_cast<const itype *>(&b_col[z].u32);
```

A_buf_의 크기:
- logical: `tile_m_ x tile_k_`
- storage: `tile_m_ x 8 words` (왜냐면 k는 항상 8 words (fp16 = 16개; fp32 = 8개))

B_buf_의 크기:
- logical: `tile_k_ x tile_n_`
- storage: `8 words x tile_n_`

**accumulator buffer 구현방식**
* 이건 `uint8_t`로 두면 안 됨

```
std::vector<float> accum_buf_;
```


---

# 2차 확장 목표
> TBD

* hardware를 더 키울 생각이 있으면 그다음에 `N=192`, `N=256`까지 확장
	* `64x256x16`은 **64KB accumulator**가 필요


### Auto selection helper

* runtime/library helper를 하나 두고 자동으로 정해주는 것
```
choose_dtcu_shape(M, N, K, fmt_s, accum_capacity, policy)
```

1. `tileK = 8 * (4 / in_sz)` 계산
2. candidate `N` set 생성
3. `problemN % tileN == 0` 필터
4. `problemK % tileK == 0` 필터
5. `accum_bytes = 64 * tileN * 4 <= accum_capacity` 필터
6. preferred set이면 16-step 우선
7. 가장 큰 legal `N` 선택
