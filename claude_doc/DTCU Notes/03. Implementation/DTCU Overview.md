
목차

- [[#Abstract|Abstract]]
- [[#Descriptor|Descriptor]]
- [[#Tile Size|Tile Size]]
- [[#Memory Request Calculation|Memory Request Calculation]]
- [[#Execution (WMMA)|Execution (WMMA)]]
- [[#WMMA|WMMA]]
- [[#Synchronization Support|Synchronization Support]]
- [[#Software Pipelining|Software Pipelining]]
- [[#Executing Matrix Bigger than Native Tile|Executing Matrix Bigger than Native Tile]]
- [[#Increasing Tile Size|Increasing Tile Size]]

# Abstract
> State machine을 돌아가며 DTCU 작동 구현
> Wait 상태는 SimX의 캐시 방식(딜레이 모델링)을 서포트하기 위하여 추가함

- `dtensor_start(desc_addr)`로 들어온 descriptor를 인자로,
- descriptor를 읽어서 `A/B/C/D` 주소, leading dimension, format, `M/N/K`, flags를 해석
- 내부 tile loop를 돌면서 operand를 메모리에서 읽고,
- 내부 `WMMA`와 동일한 fragment/FMA path로 계산하고,
- 결과를 다시 메모리에 저장

**실제 State Machine 작동 방식**
1. `IDLE`
2. `DESC_REQ`- descriptor용 representative `MemReq` 발행 ([[#Descriptor]]) ([[#Multi-tile Support]])
3. `DESC_WAIT` - 응답 오면 **실제 descriptor 값은 `ram_->read()`로 직접 읽음** ([[]])
4. `OP_REQ` - operand용 cache-line request들을 순차 발행 ([[#Memory Request Calculation]])
5. `OP_WAIT` - 응답이 다 오면 **실제 operand 값은 `ram_->read()`로 직접 읽음**
6. `EXECUTE` - 내부 fragment compute 수행
7. `OUT_REQ` - output용 cache-line request들을 순차 발행 ([[#Memory Request Calculation]])
8. `OUT_WAIT` - 끝나면 **실제 결과 값은 `ram_->write()`로 직접 저장**
9. tile loop가 남아 있으면 반복, 끝나면 `done_=true`

[[#Tile Size]]  [[#Synchronization Support]]


---
# Descriptor
**sim/simx/d_tensor_core.h:29-50**
- `ptrA`, `ptrB`, `ptrC`, `ptrD` - Device memory base address
- `ldmA`, `ldmB`, `ldmC`, `ldmD` - Leading dimension
- `M`, `N`, `K` - Matrix size
- `fmt_s`, `fmt_d` - source / destination(accumulator) element type
- `flags` - Flag for using `ptrC` or not
	- `flags & 0x1`이면 C를 읽지 않고 accumulator를 0으로 시작
	- 아니면 C를 읽어서 누산 시작

= Exactly 64B (`static_assert(sizeof(Desc) == 64)`)

## Reading Descriptor
- `tile_ptrA_()` : Row-major
- `tile_ptrB_()` : Column-major
- `tile_ptrC_()` : Row-major
- `tile_ptrD_()` : Row-major

루프를 돌아가면 ldm을 사용하여 각 major에 맞는 형식으로 element들을 불러옴

([[DTCU Basic Theory#Row / Column Major]] 참조)

#descriptor

---
# Tile Size
> Tile 사이즈는 `tensor_cfg.h`의 `wmma_config_t`에서 파생.

([[DTCU Basic Theory#Thread 갯수에 따라 Tile값이 달라지는 이유]] 참조)

### M / N Size
기본 config: `NUM_THREADS = 4` `NUM_WARPS = 4` ( `hw/VX_config.toml` 참조)

`wmma_config_t`
* 기본 tile capacity가 `NT * NR = 4 * 8 = 32`
	* NR is register per fragment
* output aspect ratio가 `tileM : tileN = 2 : 1`로 잡히기 때문에
	- `tileM = 8`
	- `tileN = 4`

### K Size
`tileK`는 input element size에 따라 달라짐.
- fp32 input이면 `tileK = 4`
- fp16/bf16 input이면 `i_ratio = 4 / 2 = 2`라서 `tileK = 8`

**따라서 regression test처럼 fp16/bf16 → fp32라면 단일 DTCU tile은 `8 x 4 x 8`**

Currently working on increasing tile size [[DTCU Increasing Tile Size]]

#tile_size

---
# Memory Request Calculation
> SimX 구조 상 L2캐시에서 데이터를 주는게 아니라 메모리에서 가져와야됨
> L1/L2는 latency modelling용 구조체
> 따라서 L2 line 크기를 기준으로 실제 사용될 만큼의 cachine line (64B) 수만큼 MemReq를 순차 발행
> DTCU 내부 coalescer를 사용

## Cache System on SimX
- SimX는 시뮬레이터이기 때문에 memory mechanism이 다름
	- 통상적인 L1/L2 -> 코어가 아니라,
	- 메모리 -> 코어로 바로 들어옴
	- 이때 latency를 모델링하기 위해 사용되는게 cache_sim 상 구현된 L1/L2

- L2 cache (cache_sim.cpp) does not actually transfer data to cores
	  **No actual payload from L2
	- L2 is acctually for modelling transaction history
		- read/write count
		- cache hit/miss count
		-  # of delayed cycles
- `mem_rsp_in` is a signal that tells “event is complete”

- **Actual data** is from emulator.cpp
	- When the SIMT core executes LD, SimX does the following:
		- **Timing Path**
			- LSU creates `MemReq`, puts into L1/L2(cachesim)
			- After a few cycle, issues `MemRsp`
				- This mimics the delay between hierarchy
		- **Functional Data Path**
			- The actual memory data that LD has to bring is from memory
				- `Emulator::dcache_read
				- `mmu_.read()
    
> cache_sim은 transaction 히스토리(req/res 수, hit/miss, 시간)를 기록하고, 실제 코어에 데이터는 emulator.cpp가 RAM/MMU에서 읽어서 줌

- Direct access to RAM’s physical address    
	- If kernel passes virtual address, it may break


## DTCU Implementation

TBD




1. A/B/C/D의 각 element 주소를 계산
2. 현재 RAM access granularity를 `WORD_BYTES = 4`로 고려
3. 각 word access가 포함되는 cache line base를 계산
	- `line_base = addr & ~(L2_LINE_SIZE - 1)`
4. 만약 word가 line boundary를 넘으면 양쪽 line 모두 포함 (결과적으로 2개의 cache line req.)
5. `std::set<uint64_t>`에 넣어서 unique한 라인만 남김
	1. 여러 데이터가 중복되는 cache line에 있으면 필요 이상으로 request를 할 수 있기 때문에
	2. set 구현체 사용함
6. unique line들에 대해서만 representative `MemReq`를 순차 발행

`line_base()` (`883-885`) - 각 element 주소 계산 helper
`coalesce_to_lines()` (`887-896`) - 중복되는 cache line 정리 helper
`build_req_lists_()` (`898-1023`)
* operand request list는 `A`, `B`, (필요시) `C`
- output request list는 `D`
- `tick()`에서:
	- `DESC_REQ`: descriptor 1 line
	- `OP_REQ`: operand lines 순차 발행
	- `OUT_REQ`: output lines 순차 발행


***Is this Physical Address or Virtual Address?***
> Physical임

* `VM_ENABLE`을 켰을 때는 host/runtime 레벨과 DTCU 내부 동작이 서로 다르게 움직임
* 기본 설정에선 VM이 꺼져있음 
	* `hw/VX_config.toml`에서 `VM_ENABLE = false`
* 따라서 memory를 allocate하는 함수 `vx_mem_alloc`는 physical address값으로 리턴
	* translation 없이 RAM에 접근
	* dtcu_basic과 dtcu_compare에서 이거 사용


### Mem_Req을 보내고 Mem_Rsp를 받기까지 기다리는 과정

- `tick()`에 `Mem_Rsp_In`이 있음
	- `sim/simx/processor.cpp` 144-145
		- 매 cycle `SimPlatform::instance().tick()`을 호출 
		- `sim/common/simobject.h` 494-500행에서 그 `tick()`이 등록된 **모든 SimObject의 `do_tick()`**을 돌림  
- DTCU는 `sim/simx/cluster.cpp` 64-65행에서 `DTensorCore::Create(...)`로 생성된 SimObject이므로, `Cluster::tick()`이 비어 있어도 `DTensorCore::tick()`은 매 cycle 호출

그래서 사용하신 말 그대로 **“mem_rsp_in이 tick에 있는데 tick을 안 써서 안 기다린다”** 는 건 아닙니다.

---
# 구 Execution (WMMA)

현재 `dtcu_basic / dtcu_compare`에선 
1. `fragA_`, `fragB_`, `fragC_`로 fragment 나눔 
	* (큰 GEMM을 타일별로 나누는거랑 다름 / 이건 [[DTCU Internal Iteration#3. 자른 tile 속에서의 계산]] 참조)
2. `rs1_data`, `rs2_data`, `rs3_data`, `rd_data` 버퍼를 준비
3. `k`, `m`, `n`에 대해 `cfg::k_steps`, `cfg::m_steps`, `cfg::n_steps` 만큼 iterate
4. 각 step마다 fragment data를 넣는다
5. `select_FEDP(fmt_s, fmt_d)`를 call함
6. 이 안에서 format별 FMA/FEDP path를 타고 결과를 계산
7. 결과를 다시 `fragC_`에 누적
8. 마지막에 `store_output()`이 `fragC_`를 D로 write

#fsm #execution

## Accumulation

***ptrC와 fragC는 다르니 조심할 것. ptrC를 설명하는건 [[DTCU Basic Theory]] 참조

같은 Output Tile에 대해 K가 여러 tile로 나뉘면, 대략 이렇게 동작:
- 맨 처음 K tile:
	- if (flags & 0x1) fragC_ = 0
	- 만약 ptrC가 지정되있다면 fragC_ = ptrC
- 이후 K tile들:
	- 계속 fragC_ 위에 누적
- 마지막에 완성된 output tile들을 각 위치에 써서 큰 matrix를 구성
	- ptrD로 저장

>  **4.4.1 Thread Block Tiling**
>  As the loop iterates, the Gemmini matrix unit ***accumulates partial sum data onto its _private accumulator memory_***, which gets moved out and stored to the global memory at the end of the loop. 

#accumulation #ptrC


---
# 신 Execution (MMA)

## 전체 구조

```C++
void DTensorCore::execute_wmma() {  
  auto fedp = select_FEDP(desc_.fmt_s, desc_.fmt_d);  
  
  if ((DTCU_TILE_K_WORDS % cfg::tcK) != 0) {  
    std::cout << "[DTCU] Error: Tile K is not divisible by FEDP width" << std::endl;  
    std::abort();  
  }  
  
  for (uint32_t m = 0; m < tile_m_; ++m) {  
    for (uint32_t n = 0; n < tile_n_; ++n) {  
      uint32_t acc_bit;  
        
      std::memcpy(&acc_bit, &accum_buf_[m * tile_n_ + n], 4);  
  
      for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; kw += cfg::tcK) {  
        std::array<reg_data_t, cfg::tcK> a_words{};  
        std::array<reg_data_t, cfg::tcK> b_words{};  
  
        for (uint32_t z = 0; z < cfg::tcK; ++z) {  
          a_words[z].u32 = a_buf_[m * DTCU_TILE_K_WORDS + kw + z];  
          b_words[z].u32 = b_buf_[(kw + z) * tile_n_ + n];  
        }  
  
        acc_bit = fedp(a_words.data(), b_words.data(), acc_bit);  
      }  
  
      std::memcpy(&accum_buf_[m * tile_n_ + n], &acc_bit, 4);  
    }  
  }  
}
```

### select_FEDP(desc_.fmt_s, desc_.fmt_d)`
```
auto fedp = select_FEDP(desc_.fmt_s, desc_.fmt_d);
```

input/output datatype 조합에 맞는 dot-product backend를 골라줌

예를 들어,
- fp16 -> fp32 이면 `FEDP<vt::fp16, vt::fp32>::eval`
- bf16 -> fp32 이면 `FEDP<vt::bf16, vt::fp32>::eval`
- fp32 -> fp32 이면 그 조합에 맞는 eval

### DTCU_TILE_K_WORDS % cfg::tcK == 0`
> K 방향 packed word 8개를 `fedp`가 요구하는 micro-width 단위로 정확히 쪼갤 수 있어야 하기 때문

`fedp`는 호출되면 `cfg::tcK`개의 `reg_data_t`를 받음. 전체 K word 폭이 8이면.
- `cfg::tcK`
- `cfg::tcK`
- `cfg::tcK`
- ...
이렇게 나눠져야됨. 

예를 들어,
- `DTCU_TILE_K_WORDS = 8`
- `cfg::tcK = 2`
면 `kw = 0, 2, 4, 6` 총 4번 돌게됨

따라서 **새 dense buffer layout과 기존 FEDP micro-kernel 폭이 호환되는지** 확인해야됨


### Outer Loop - M/N Iterate
> 전체를 element 별로 처리

```
for (uint32_t m = 0; m < tile_m_; ++m) {  
  for (uint32_t n = 0; n < tile_n_; ++n) {
```
이 두 loop는 output tile의 모든 element를 순회합니다.

즉 `(m,n)` 하나는
- output tile 현재 m번째 row
- output tile 현재 n번째 column

새 버전은 lane을 순회하지 않습니다.  
원본은 lane별 fragment를 조립해서 output sub-block을 계산했지만,  
새 버전은 **바로 output element space에서 순회**합니다.

이게 thread dependency를 끊은 핵심입니다.

### accum_buf_[m * tile_n_ + n] 를 읽는 이유
```
uint32_t acc_bit;  
std::memcpy(&acc_bit, &accum_buf_[m * tile_n_ + n], 4);
```

**옵션 1: 첫 K tile**
`load_operands()`에서
- `flags & 0x1`이면 zero-init
- 아니면 C tile을 읽어옴

따라서 초기 accumulator 값

**옵션 2: 두 번째 이후 K tile**
이전 `execute_wmma()`에서 이미 partial sum이 누적

따라서 **현재까지의 partial accumulator**

`fedp`에 넘기려면 raw 32-bit bit pattern이 필요하므로 `memcpy`로 읽음.


### Internal Loop - Iterate K 
> 전체를 micro-step(kw)으로 나누어 처리

```
for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; kw += cfg::tcK) {
```

FEDP 입력 단위는 element가 아니라 32-bit packed word
* 따라서 내부적으로 K를 32-bit packed word 만큼 잘라서 받아야됨
	* K 전체를 `cfg::tcK` word씩 잘라서 처리
- fp32 input이면 word 하나에 element 1개
- fp16/bf16 input이면 word 하나에 element 2개



---

## 9. `a_words`, `b_words`를 왜 새로 만들나

std::array<reg_data_t, cfg::tcK> a_words{};  
std::array<reg_data_t, cfg::tcK> b_words{};

이건 `fedp`가 요구하는 입력 형식이기 때문입니다.

원본 `FEDP::eval()` signature가:

uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val)

이므로, `fedp`는 `cfg::tcK` 길이의 `reg_data_t` 배열 두 개를 받습니다.

즉 이 배열은

- A row의 현재 micro-slice
- B column의 현재 micro-slice

를 temporary하게 만들어서 넘기는 용도입니다.

---

## 10. A gather: `a_buf_[m * DTCU_TILE_K_WORDS + kw + z]`

a_words[z].u32 = a_buf_[m * DTCU_TILE_K_WORDS + kw + z];

이건 A에서 **m번째 row**를 읽고 있습니다.

buffer shape를 다시 쓰면 `a_buf_`는:

[tile_m_][DTCU_TILE_K_WORDS]

입니다.

따라서 index:

m * DTCU_TILE_K_WORDS + (kw + z)

는

- row = `m`
- K-word position = `kw + z`

입니다.

즉 A의 m번째 row에서 현재 micro-slice에 해당하는 packed word들을 `a_words`에 채우는 것입니다.

---

## 11. B gather: `b_buf_[(kw + z) * tile_n_ + n]`

b_words[z].u32 = b_buf_[(kw + z) * tile_n_ + n];

이건 B에서 **n번째 column**을 읽고 있습니다.

buffer shape가:

[DTCU_TILE_K_WORDS][tile_n_]

이므로,

(kw + z) * tile_n_ + n

는

- row in K-words = `kw + z`
- column = `n`

입니다.

즉 B의 n번째 column에서 현재 micro-slice의 packed word들을 모읍니다.

---

## 12. 왜 A는 row-major gather이고 B는 column gather처럼 보이나

이건 matrix multiply 자체가

D[m,n] += sum_k A[m,k] * B[k,n]

이기 때문입니다.

출력 원소 `(m,n)`를 계산하려면 필요한 것은:

- A의 **m번째 row**
- B의 **n번째 column**

입니다.

그래서 새 `execute_wmma()`는 정확히 그 수학 형태대로 데이터를 꺼냅니다.

---

## 13. `fedp(a_words.data(), b_words.data(), acc_bit)`가 실제로 하는 일

acc_bit = fedp(a_words.data(), b_words.data(), acc_bit);

이 한 줄이 실제 dot-product micro-kernel입니다.

내부적으로는:

1. `acc_bit`를 output type으로 해석
2. `z = 0 .. cfg::tcK-1` 반복
3. 각 `a_words[z].u32`, `b_words[z].u32`를 input type 배열로 reinterpret
4. word 내부 element들을 모두 multiply-add
5. 다시 raw 32-bit로 반환

합니다.

즉 `fedp` 1회는 **K direction의 `cfg::tcK` words**를 처리합니다.

실제 element 수로는:

cfg::tcK * (4 / sizeof(input_type))

입니다.

예를 들어 fp16이면 `i_ratio = 2`라서,  
word 하나당 fp16 두 개가 있으므로 `fedp` 한 번에 더 많은 logical K element를 처리합니다.

---

## 14. `kw` loop 전체가 끝나면 무엇이 되나

`kw` loop는 `0 -> cfg::tcK -> ... -> 8-cfg::tcK` 식으로 돌면서  
K direction 전체 8 words를 모두 처리합니다.

따라서 `kw` loop가 끝났다는 것은:

- A row `m`
- B column `n`
- tile 내 K 전체

에 대한 reduction이 다 끝났다는 뜻입니다.

즉 이 시점의 `acc_bit`는 output tile element `(m,n)`의 최종 누적값입니다.  
정확히는 “이번 K tile까지 반영한 누적값”입니다.

---

## 15. 왜 마지막에 다시 `accum_buf_`에 써넣나

std::memcpy(&accum_buf_[m * tile_n_ + n], &acc_bit, 4);

이건 이번 `(m,n)` 계산 결과를 accumulator buffer에 반영하는 것입니다.

그래서 다음 K tile이 오면, `load_operands()`가 zero-init하지 않는 한  
이 값 위에 계속 누적됩니다.

즉 `accum_buf_`는 단순 output scratch가 아니라 **K tiles across accumulation state**입니다.

---

## 16. 이 함수 전체를 수학식으로 쓰면

이 함수는 결국 각 `(m,n)`에 대해 다음을 합니다.

accum_buf_[m,n] =  
  accum_buf_[m,n] +  
  Σ_k ( A_tile[m,k] * B_tile[k,n] )

단, 이 `Σ_k`를 한 번에 하지 않고,

- packed 32-bit word 기준으로 K를 자르고
- 그 조각들을 `fedp`에 여러 번 넣어
- 누적하는 방식입니다.

즉 수학적으로는 GEMM과 같고, 구현적으로는:

- **outer loop**: output element `(m,n)`
- **inner loop**: K direction micro-step
- **micro-kernel**: `fedp`



---
# Synchronization Support
> Barrier와 fence를 통해 synchronization 구현 (kernel에서 구현)
> "Collaborative execution + barrier + fence/busy polling"

Virgo에서는 여러 core가 각자 DTCU에 command를 동시에 던지고, DTCU가 queueing해서 처리하는 모델이 아님

**여러 core/warp가 하나의 matrix operation을 위해 협업**하고, 그 전에 필요한 **작업들(descriptor 준비, data movement, post-processing 등)을 분산해서 수행**한 뒤, cluster-wide barrier로 시점을 맞추고, **대표 쪽이 matrix unit launch/control을 담당**

(Barrier에 관한 내용은 [[DTCU Basic Theory#Barrier]] 참조)

- Concurrent start handling
	- Currently ignores when state is busy_
- There already is barrier substrate
	- vx_barrier, async barrier, cluster barrier aggregation

현재 구현 방식은 launch 자체보다 “누가 launch를 책임지고, 나머지는 언제 barrier/fence로 합류하느냐가 중점

1. 여러 core가 descriptor/data 준비 
2. vx_fence()
3. vx_barrier() (Global Barrier)
4. Leading core issues dtensor_start()
5. Leading core issues dtensor_poll()
6. When completed, vx_fence()
7. Global barrier
8. 다른 core들도 다음 단계 진행

---
# Software Pipelining
### 지금 한 것

- **kernel에서 barrier/fence를 호출한 synchronization**

### 나중에 더 할 것
- **kernel에서 loop/control flow로 구현**
- 여러 쓰레드가 같이 준비 
- asynchronous barrier 사용

**software pipelining은 아직 아님.**


### Idea 써놓는 곳
* Output tile 하나 연산하고 메모리에 쓸 때
	* DMA (또는 쓰레드 하나)가 동시에 메모리로 씀
	* DTCU는 다음 output tile로 바로 넘어감
	* 이럼 output tile 쓰는걸 기다릴 필요가 없음
		*  out_wait 필요가 없어짐

---
# Executing Matrix Bigger than Native Tile

[[DTCU Internal Iteration]]

---
# Increasing Tile Size

[[DTCU Increasing Tile Size]]